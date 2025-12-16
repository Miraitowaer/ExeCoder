import os
import torch
import torch.nn.functional as F
import multiprocessing
import subprocess
import tempfile
import re
import logging
import time
import gc
import difflib # 仅用于选择 Hard Negative，不再用于 Mask
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Accelerate & Distributed
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DummyOptim

# Transformers & Data
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    HfArgumentParser
)
from tqdm import tqdm

# ================= 全局配置 =================
IGNORE_INDEX = -100
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ScriptArguments:
    sft_model_path: str = field(metadata={"help": "SFT 模型路径"})
    data_path: str = field(metadata={"help": "训练数据路径 (JSON)"})
    output_dir: str = field(metadata={"help": "输出目录"})
    
    # 采样配置
    num_generations: int = field(default=2, metadata={"help": "推荐设为2以加速"})
    temperature: float = field(default=0.8)
    max_new_tokens: int = field(default=384)
    
    # 训练配置
    learning_rate: float = field(default=5e-7)
    per_device_train_batch_size: int = field(default=1) 
    gradient_accumulation_steps: int = field(default=8)
    num_train_epochs: int = field(default=1)
    beta: float = field(default=0.1)
    max_length: int = field(default=2048)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500) 

# ================= 模块 A: 执行器 (保持不变) =================
class CodeExecutor:
    @staticmethod
    def detect_language(instruction: str, gt_code: str) -> str:
        if "```cpp" in gt_code or "```c++" in gt_code: return "cpp"
        if "```python" in gt_code: return "python"
        if "```java" in gt_code: return "java"
        instruction = instruction.lower()
        if " to cpp" in instruction or " to c++" in instruction: return "cpp"
        if " to java" in instruction: return "java"
        if " to python" in instruction: return "python"
        return "cpp" 

    @staticmethod
    def run_code_sync(code: str, lang: str, input_str: str = "") -> Tuple[bool, str]:
        code = re.sub(r'```[a-zA-Z]*', '', code).replace('```', '').strip()
        is_success = False
        output_str = ""
        try:
            if lang in ['python', 'py']:
                with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
                    f.write(code)
                    file_path = f.name
                proc = subprocess.run(['python3', file_path], input=input_str, text=True, capture_output=True, timeout=3)
                if proc.returncode == 0:
                    is_success = True; output_str = proc.stdout.strip()
                os.remove(file_path)
            elif lang in ['cpp', 'c++']:
                with tempfile.NamedTemporaryFile(suffix=".cpp", mode='w', delete=False) as f:
                    f.write(code)
                    src_path = f.name
                bin_path = src_path + ".out"
                comp = subprocess.run(['g++', '-O2', '-w', src_path, '-o', bin_path], capture_output=True, text=True)
                if comp.returncode == 0:
                    try:
                        proc = subprocess.run([bin_path], input=input_str, text=True, capture_output=True, timeout=2)
                        if proc.returncode == 0: is_success = True; output_str = proc.stdout.strip()
                    except subprocess.TimeoutExpired: pass
                    if os.path.exists(bin_path): os.remove(bin_path)
                if os.path.exists(src_path): os.remove(src_path)
            elif lang == 'java':
                class_name_match = re.search(r'class\s+(\w+)', code)
                class_name = class_name_match.group(1) if class_name_match else "Main"
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, f"{class_name}.java")
                    with open(file_path, 'w') as f: f.write(code)
                    comp = subprocess.run(['javac', file_path], capture_output=True, text=True)
                    if comp.returncode == 0:
                        try:
                            proc = subprocess.run(['java', '-cp', temp_dir, class_name], input=input_str, text=True, capture_output=True, timeout=3)
                            if proc.returncode == 0: is_success = True; output_str = proc.stdout.strip()
                        except subprocess.TimeoutExpired: pass
        except Exception: pass 
        return is_success, output_str

    def batch_evaluate(self, gt_code: str, candidates: List[str], lang: str) -> Tuple[List[str], List[str]]:
        gt_success, gt_stdout = self.run_code_sync(gt_code, lang)
        if not gt_success: return [], []
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = [executor.submit(self.run_code_sync, code, lang) for code in candidates]
            results = [f.result() for f in futures]
        pass_list, fail_list = [], []
        for cand, (succ, out) in zip(candidates, results):
            if succ and out.strip() == gt_stdout.strip(): pass_list.append(cand)
            else: fail_list.append(cand)
        return pass_list, fail_list

# ================= 模块 B: 标准 DPO 处理器 (Standard DPO) =================
# [Change]: 替换了原来的 MaskProcessor
class StandardDPOProcessor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_pair(self, prompt: str, chosen: str, rejected: str):
        """
        标准 DPO Tokenization:
        Label 在 Prompt 部分为 -100，在 Response 部分为 Token ID
        """
        # 1. 拼接文本
        c_full = prompt + chosen
        r_full = prompt + rejected
        
        # 2. 编码 (Truncation enabled)
        c_enc = self.tokenizer(c_full, truncation=True, max_length=self.max_length, return_offsets_mapping=True, return_tensors='pt')
        r_enc = self.tokenizer(r_full, truncation=True, max_length=self.max_length, return_offsets_mapping=True, return_tensors='pt')
        
        prompt_len = len(prompt)
        
        def process_one(enc):
            input_ids = enc.input_ids[0]
            labels = input_ids.clone()
            offsets = enc.offset_mapping[0]
            
            # 找到 Prompt 结束的位置
            # 任何 offset 的 start < prompt_len 的都属于 prompt (大致逻辑)
            resp_start_idx = 0
            for i, (s, e) in enumerate(offsets):
                if s >= prompt_len:
                    resp_start_idx = i
                    break
            
            # Mask Prompt (设为 -100)
            labels[:resp_start_idx] = IGNORE_INDEX
            
            return input_ids, enc.attention_mask[0], labels

        c_ids, c_mask, c_lbl = process_one(c_enc)
        r_ids, r_mask, r_lbl = process_one(r_enc)
        
        return c_ids, c_mask, c_lbl, r_ids, r_mask, r_lbl

# ================= 模块 C: 训练逻辑 (标准 DPO 计算) =================
def compute_dpo_loss(model, ref_model, batch, beta=0.1):
    policy_logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
    with torch.no_grad():
        ref_logits = ref_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits

    labels = batch['labels']
    logits_p = policy_logits[:, :-1, :]
    logits_r = ref_logits[:, :-1, :]
    labels_shifted = labels[:, 1:]
    
    # 将 -100 的位置置为 0，防止 gather 越界
    safe_labels = labels_shifted.clone()
    safe_labels[safe_labels == IGNORE_INDEX] = 0
    
    per_token_logps_p = torch.gather(logits_p.log_softmax(-1), 2, safe_labels.unsqueeze(2)).squeeze(2)
    per_token_logps_r = torch.gather(logits_r.log_softmax(-1), 2, safe_labels.unsqueeze(2)).squeeze(2)
    
    # 只有 Response 部分才有 Loss (labels != -100)
    valid_mask = (labels_shifted != IGNORE_INDEX).float()
    
    policy_logps = (per_token_logps_p * valid_mask).sum(-1)
    ref_logps = (per_token_logps_r * valid_mask).sum(-1)
    
    chosen_policy = policy_logps[::2]
    rejected_policy = policy_logps[1::2]
    chosen_ref = ref_logps[::2]
    rejected_ref = ref_logps[1::2]
    
    # DPO Loss
    chosen_rewards = beta * (chosen_policy - chosen_ref)
    rejected_rewards = beta * (rejected_policy - rejected_ref)
    
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    margins = chosen_rewards - rejected_rewards
    
    loss = -F.logsigmoid(margins).mean()
    
    metrics = {
        "rewards/chosen": chosen_rewards.mean().item(),
        "rewards/rejected": rejected_rewards.mean().item(),
        "rewards/accuracies": reward_accuracies.mean().item(),
        "rewards/margins": margins.mean().item(),
        "logps/chosen": chosen_policy.mean().item(),
        "logps/rejected": rejected_policy.mean().item()
    }
    
    return loss, metrics

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(42)
    
    # [Config]: 保持 3 小时超时，确保稳定
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=180))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        log_with="tensorboard", 
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        accelerator.init_trackers(project_name="runs", config=vars(args))
        logger.info(f"Initialized STANDARD Online DPO Trainer (No Mask). Output: {args.output_dir}")

    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    # 显存优化
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False 
    
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model = ref_model.to(accelerator.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    optimizer = DummyOptim(model.parameters(), lr=args.learning_rate)
    
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    executor_tool = CodeExecutor()
    # [Change]: 使用标准 DPO Processor
    processor = StandardDPOProcessor(tokenizer, args.max_length)
    
    global_step = 0
    model.train()
    
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch_data in progress_bar:
            # === Phase 1: Mining (生成) ===
            instructions = batch_data['instruction']
            inputs = batch_data['input']
            outputs = batch_data['output']
            
            batch_input_ids, batch_labels, batch_masks = [], [], []
            mined_count = 0
            
            torch.cuda.empty_cache() 
            
            for i, instr in enumerate(instructions):
                gt_code_raw = outputs[i]
                target_lang = CodeExecutor.detect_language(instr, gt_code_raw)
                inp_data = inputs[i]
                
                if inp_data: prompt = ALPACA_PROMPT_DICT["prompt_input"].format(instruction=instr, input=inp_data)
                else: prompt = ALPACA_PROMPT_DICT["prompt_no_input"].format(instruction=instr)
                
                with torch.no_grad():
                    inputs_tokenized = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
                    # 临时开启 Cache 加速生成
                    gen_ids = accelerator.unwrap_model(model).generate(
                        **inputs_tokenized,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        num_return_sequences=args.num_generations,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True 
                    )
                
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                candidates = []
                for text in gen_texts:
                    parts = text.split("### Response:")
                    c = parts[1] if len(parts) > 1 else text
                    c = re.sub(r'```[a-zA-Z]*', '', c).replace('```', '').strip()
                    candidates.append(c)
                
                gt_code = re.sub(r'```[a-zA-Z]*', '', gt_code_raw).replace('```', '').strip()
                passes, fails = executor_tool.batch_evaluate(gt_code, candidates, target_lang)
                
                if not fails: continue 
                
                chosen = passes[0] if passes else gt_code
                # 依然使用 Hard Negative 策略选择 Rejected，但不做 Diff Mask
                rejected = max(fails, key=lambda x: difflib.SequenceMatcher(None, chosen, x).ratio())
                if chosen == rejected: continue
                
                # === Phase 2: Standard Tokenization (无 Mask) ===
                c_ids, c_mask, c_lbl, r_ids, r_mask, r_lbl = processor.tokenize_pair(prompt, chosen, rejected)
                
                batch_input_ids.extend([c_ids, r_ids])
                batch_masks.extend([c_mask, r_mask])
                batch_labels.extend([c_lbl, r_lbl])
                mined_count += 1
            
            # === Phase 3: Training (训练) ===
            torch.cuda.empty_cache()
            gc.collect()

            # DDP Sync Logic
            if mined_count > 0:
                max_len = max([t.size(0) for t in batch_input_ids])
                padded_ids = [F.pad(t, (0, max_len-t.size(0)), value=tokenizer.pad_token_id) for t in batch_input_ids]
                padded_masks = [F.pad(t, (0, max_len-t.size(0)), value=0) for t in batch_masks]
                padded_labels = [F.pad(t, (0, max_len-t.size(0)), value=IGNORE_INDEX) for t in batch_labels]
                
                final_batch = {
                    'input_ids': torch.stack(padded_ids).to(accelerator.device),
                    'attention_mask': torch.stack(padded_masks).to(accelerator.device),
                    'labels': torch.stack(padded_labels).to(accelerator.device)
                }
                
                with accelerator.accumulate(model):
                    loss, metrics = compute_dpo_loss(model, ref_model, final_batch, beta=args.beta)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if global_step % args.logging_steps == 0:
                    logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | Acc: {metrics['rewards/accuracies']:.2f} | Mined: {mined_count}")
                    if accelerator.is_main_process:
                        log_data = {
                            "train/loss": loss.item(),
                            "train/mined_pairs": mined_count,
                            "train/epoch": epoch,
                        }
                        log_data.update(metrics)
                        accelerator.log(log_data, step=global_step)
            else:
                # Dummy Sync
                dummy_input = tokenizer(instructions[0], return_tensors="pt").input_ids.to(accelerator.device)
                if dummy_input.size(1) > 10: dummy_input = dummy_input[:, :10]
                
                with accelerator.accumulate(model):
                    dummy_out = model(dummy_input)
                    dummy_loss = dummy_out.logits.mean() * 0.0
                    accelerator.backward(dummy_loss)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if global_step % args.logging_steps == 0:
                    logger.info(f"Step {global_step} | Loss: 0.0000 (Dummy Sync) | Mined: 0")

            global_step += 1
            
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Checkpoint saved to {save_path}")
    
    if accelerator.is_main_process:
        accelerator.save_state(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        accelerator.end_training()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()