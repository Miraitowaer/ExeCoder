import os
import torch
import difflib
import multiprocessing
import subprocess
import tempfile
import random
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from torch.utils.data import IterableDataset, DataLoader

# ================= 1. 全局配置与模板 =================
IGNORE_INDEX = -100

# 适配 SFT 格式
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
    data_path: str = field(metadata={"help": "SFT 训练数据路径 (input/output/instruction)"})
    lang: str = field(default="cpp", metadata={"help": "目标语言: cpp/python/java"})
    num_generations: int = field(default=4, metadata={"help": "每次采样数量"})
    temperature: float = field(default=0.8, metadata={"help": "采样温度"})

# ================= 2. 掩码逻辑 (不变) =================
def get_dual_diff_ranges(chosen_code: str, rejected_code: str):
    chosen_lines = chosen_code.splitlines(keepends=True)
    rejected_lines = rejected_code.splitlines(keepends=True)
    
    def get_line_offsets(lines):
        offsets = []
        pos = 0
        for line in lines:
            offsets.append((pos, pos + len(line)))
            pos += len(line)
        return offsets
        
    c_offsets = get_line_offsets(chosen_lines)
    r_offsets = get_line_offsets(rejected_lines)
    matcher = difflib.SequenceMatcher(None, chosen_lines, rejected_lines)
    c_ranges, r_ranges = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal': continue 
        elif tag == 'replace':
            if i1 < len(c_offsets): c_ranges.append((c_offsets[i1][0], c_offsets[min(i2-1, len(c_offsets)-1)][1]))
            if j1 < len(r_offsets): r_ranges.append((r_offsets[j1][0], r_offsets[min(j2-1, len(r_offsets)-1)][1]))
        elif tag == 'insert':
            if j1 < len(r_offsets): r_ranges.append((r_offsets[j1][0], r_offsets[min(j2-1, len(r_offsets)-1)][1]))
        elif tag == 'delete':
            if i1 < len(c_offsets): c_ranges.append((c_offsets[i1][0], c_offsets[min(i2-1, len(c_offsets)-1)][1]))
            if j1 < len(r_offsets): r_ranges.append((r_offsets[j1][0], r_offsets[j1][1]))
            
    return c_ranges, r_ranges

def apply_mask_logic(input_ids, offset_mapping, focus_ranges, prompt_len):
    labels = list(input_ids)
    for i in range(len(labels)): labels[i] = IGNORE_INDEX
        
    response_start_idx = 0
    for i, (start, end) in enumerate(offset_mapping):
        if start >= prompt_len:
            response_start_idx = i
            break
            
    for idx in range(response_start_idx, len(offset_mapping)):
        start, end = offset_mapping[idx]
        if start == 0 and end == 0: continue
        token_start, token_end = start - prompt_len, end - prompt_len
        
        is_focus = False
        for (fs, fe) in focus_ranges:
            if max(token_start, fs) < min(token_end, fe):
                is_focus = True
                break
        if is_focus: labels[idx] = input_ids[idx]
            
    return labels

# ================= 3. 新的执行逻辑：Run & Capture Stdout =================
def run_code_and_capture_output(args):
    """
    编译运行代码，并返回 (Success, Stdout)
    """
    code, lang = args
    
    output_str = ""
    is_success = False

    try:
        if lang.lower() in ['python', 'py']:
            with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
                f.write(code)
                file_path = f.name
            
            # 直接运行，不输入 stdin，因为你的代码里已经 hardcode 了数据
            proc = subprocess.run(['python3', file_path], text=True, capture_output=True, timeout=3)
            if proc.returncode == 0:
                is_success = True
                output_str = proc.stdout.strip()
            
            os.remove(file_path)

        elif lang.lower() in ['cpp', 'c++']:
            with tempfile.NamedTemporaryFile(suffix=".cpp", mode='w', delete=False) as f:
                f.write(code)
                src_path = f.name
            bin_path = src_path + ".out"
            
            # 编译
            comp = subprocess.run(['g++', '-O2', src_path, '-o', bin_path], capture_output=True)
            if comp.returncode == 0:
                # 运行
                try:
                    proc = subprocess.run([bin_path], text=True, capture_output=True, timeout=2)
                    if proc.returncode == 0:
                        is_success = True
                        output_str = proc.stdout.strip()
                except subprocess.TimeoutExpired:
                    is_success = False
                
                if os.path.exists(bin_path): os.remove(bin_path)
            
            if os.path.exists(src_path): os.remove(src_path)

    except Exception:
        is_success = False

    return (is_success, output_str)

def execute_batch_compare(gt_code, candidate_codes, lang):
    """
    1. 跑 GT 代码拿到预期输出
    2. 跑候选代码进行对比
    """
    # 1. 运行 GT (Ground Truth)
    success_gt, expected_out = run_code_and_capture_output((gt_code, lang))
    
    # 如果 GT 自己都跑不通（编译失败或超时），这条数据就废了，没法评测
    if not success_gt:
        return [False] * len(candidate_codes)

    # 2. 并行运行候选代码
    tasks = [(c, lang) for c in candidate_codes]
    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
        results = pool.map(run_code_and_capture_output, tasks)
    
    # 3. 对比输出
    pass_results = []
    for success, out in results:
        if success and out == expected_out:
            pass_results.append(True)
        else:
            pass_results.append(False)
            
    return pass_results

# ================= 4. 在线数据集 (适配无 TestCase 模式) =================
class OnlineExecutionDataset(IterableDataset):
    def __init__(self, sft_dataset, model, tokenizer, script_args):
        self.sft_dataset = sft_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.args = script_args

    def __iter__(self):
        iterator = iter(self.sft_dataset)
        while True:
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(self.sft_dataset)
                data = next(iterator)

            instr, inp, gt_code = data['instruction'], data['input'], data['output']
            
            # 清洗 GT 代码的 Markdown (用于执行)
            gt_code_clean = gt_code.replace(f"```{self.args.lang}", "").replace("```", "").strip()

            # 1. 准备 Prompt
            if inp:
                prompt = ALPACA_PROMPT_DICT["prompt_input"].format(instruction=instr, input=inp)
            else:
                prompt = ALPACA_PROMPT_DICT["prompt_no_input"].format(instruction=instr)

            # 2. 在线 Rollout
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            self.model.eval()
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **input_ids,
                    max_new_tokens=512,
                    num_return_sequences=self.args.num_generations,
                    do_sample=True,
                    temperature=self.args.temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            self.model.train()
            
            gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            codes = []
            for t in gen_texts:
                parts = t.split("### Response:")
                c = parts[1] if len(parts) > 1 else t
                c = c.replace(f"```{self.args.lang}", "").replace("```", "").strip()
                codes.append(c)
                
            # 3. Execution (动态对拍)
            # 这里的核心逻辑：先跑 GT 拿结果，再跑 Candidates 对比
            pass_results = execute_batch_compare(gt_code_clean, codes, self.args.lang)
            
            # 4. Mining
            passes = [c for c, r in zip(codes, pass_results) if r]
            fails = [c for c, r in zip(codes, pass_results) if not r]
            
            # 策略：如果模型生成的代码跑通了，就用它做 Chosen (Self-Distillation)
            # 如果没跑通，就用 SFT 数据集里的 GT 代码做 Chosen
            if passes:
                chosen = passes[0]
            else:
                # 即使没跑通，GT 也是保底
                chosen = gt_code_clean
                
            # 必须有错误样本才能构成 DPO Pair
            if not fails: continue 

            # Hard Negative: 找最像 Chosen 的 Fail
            rejected = max(fails, key=lambda x: difflib.SequenceMatcher(None, chosen, x).ratio())

            if chosen == rejected: continue

            # 5. Masking
            c_ranges, r_ranges = get_dual_diff_ranges(chosen, rejected)
            
            c_full = prompt + chosen
            r_full = prompt + rejected
            
            c_enc = self.tokenizer(c_full, truncation=True, max_length=2048, return_offsets_mapping=True)
            r_enc = self.tokenizer(r_full, truncation=True, max_length=2048, return_offsets_mapping=True)
            
            c_labels = apply_mask_logic(c_enc.input_ids, c_enc.offset_mapping, c_ranges, len(prompt))
            r_labels = apply_mask_logic(r_enc.input_ids, r_enc.offset_mapping, r_ranges, len(prompt))
            
            yield {
                "prompt": prompt,
                "chosen_input_ids": c_enc.input_ids,
                "chosen_attention_mask": c_enc.attention_mask,
                "chosen_labels": c_labels,
                "rejected_input_ids": r_enc.input_ids,
                "rejected_attention_mask": r_enc.attention_mask,
                "rejected_labels": r_labels,
            }

# ================= 5. Trainer 与 主函数 =================
class OnlineMaskDPOTrainer(DPOTrainer):
    def __init__(self, sft_dataset, script_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_dataset_source = sft_dataset
        self.script_args = script_args
        
    def get_train_dataloader(self) -> DataLoader:
        online_dataset = OnlineExecutionDataset(
            sft_dataset=self.sft_dataset_source,
            model=self.model,
            tokenizer=self.tokenizer,
            script_args=self.script_args
        )
        return DataLoader(
            online_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=True
        )

def main():
    from transformers import HfArgumentParser
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.remove_unused_columns = False # 保护 labels 不被删除
    
    print(f"Loading Dataset from {script_args.data_path}...")
    dataset = load_dataset("json", data_files=script_args.data_path, split="train")
    
    print(f"Loading Model from {script_args.sft_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    print("Initializing Online Mask-DPO Trainer...")
    trainer = OnlineMaskDPOTrainer(
        sft_dataset=dataset,
        script_args=script_args,
        model=model,
        ref_model=None, 
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=2048,
        max_prompt_length=1024,
    )
    
    print("Starting Training Loop (Generate -> Execute GT -> Execute Cand -> Mask -> Update)...")
    trainer.train()
    
    print(f"Saving to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()