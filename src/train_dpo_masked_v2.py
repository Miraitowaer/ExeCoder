import os
import difflib
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig

# ================= 全局常量 (解决 AttributeError) =================
IGNORE_INDEX = -100 
# ==============================================================

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the SFT model"})
    data_path: str = field(metadata={"help": "Path to the DPO dataset (json)"})

# ===================== 自定义双向 Masked DPO Trainer =====================
class DualMaskedDPOTrainer(DPOTrainer):
    """
    [升级版] 
    不仅 Mask Rejected 的正确部分，也 Mask Chosen 的相同部分。
    实现真正的"差异对齐 (Diff Alignment)"。
    """
    
    def get_dual_diff_ranges(self, chosen_code: str, rejected_code: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        计算 Chosen 和 Rejected 分别需要 Focus 的字符范围。
        Returns: (chosen_focus_ranges, rejected_focus_ranges)
        """
        chosen_lines = chosen_code.splitlines(keepends=True)
        rejected_lines = rejected_code.splitlines(keepends=True)
        
        # 1. 构建行号到字符位置的映射
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
        
        c_ranges = []
        r_ranges = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # [关键改变] 相同部分，两边都不训练！全部 Mask！
                continue
                
            elif tag == 'replace':
                # Chosen(i1:i2) vs Rejected(j1:j2)
                # 两边都有差异，两边都要训练
                if i1 < len(c_offsets):
                    c_ranges.append((c_offsets[i1][0], c_offsets[min(i2-1, len(c_offsets)-1)][1]))
                if j1 < len(r_offsets):
                    r_ranges.append((r_offsets[j1][0], r_offsets[min(j2-1, len(r_offsets)-1)][1]))
                    
            elif tag == 'insert':
                # Rejected 多写了 (Chosen没有)
                # Chosen: 无需操作
                # Rejected: 惩罚多写的部分
                if j1 < len(r_offsets):
                    r_ranges.append((r_offsets[j1][0], r_offsets[min(j2-1, len(r_offsets)-1)][1]))
                    
            elif tag == 'delete':
                # Chosen 有，Rejected 漏写了
                # Chosen: 奖励这部分 (Focus)
                if i1 < len(c_offsets):
                    c_ranges.append((c_offsets[i1][0], c_offsets[min(i2-1, len(c_offsets)-1)][1]))
                
                # Rejected: 惩罚断层 (Missing Penalty)
                # 惩罚 j1 行 (如果存在)
                if j1 < len(r_offsets):
                    r_ranges.append((r_offsets[j1][0], r_offsets[j1][1]))

        return c_ranges, r_ranges

def tokenize_row(self, feature: Dict[str, Any], model: Optional = None) -> Dict[str, Any]:
        """
        [SFT-Aligned Version]
        1. 注入 Alpaca 模板 (Preamble + Instruction)，确保与 SFT 输入分布一致。
        2. 执行双向 Mask (Chosen & Rejected) 逻辑。
        """
        batch = {}
        prompt_raw = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not isinstance(prompt_raw, str):
            raise ValueError(f"prompt should be an str but got {type(prompt_raw)}")
        
        # ================= Template Injection (SFT 对齐) =================
        # 定义 SFT 训练时使用的标准 Preamble (开场白)
        # 必须完全一致，包括标点符号和换行
        ALPACA_PREAMBLE = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
        )
        
        # 构造带模板的 Prompt
        if "### Response:" not in prompt_raw:
            # 如果是纯文本，加上 Preamble 和 Tag
            # 注意：我们将 prompt_raw 整体放入 ### Instruction 块中，
            # 这是处理 DPO 混合字段最稳妥的方式，关键是有 Preamble 激活模型。
            prompt = f"{ALPACA_PREAMBLE}### Instruction:\n{prompt_raw}\n\n### Response:\n"
        else:
            # 防止重复添加
            prompt = prompt_raw
        # ===============================================================
        
        # 1. Tokenize (开启 offset_mapping)
        # 注意：这里使用的是加了模板后的 prompt
        
        # Tokenize Chosen
        chosen_full_text = prompt + chosen
        chosen_tokens = self.tokenizer(
            chosen_full_text, truncation=True, max_length=self.max_length, padding=False, return_offsets_mapping=True
        )
        
        # Tokenize Rejected
        rejected_full_text = prompt + rejected
        rejected_tokens = self.tokenizer(
            rejected_full_text, truncation=True, max_length=self.max_length, padding=False, return_offsets_mapping=True
        )

        # 2. 获取 Diff Ranges (双向)
        # Diff 是基于纯代码内容 (Response) 计算的，与 Prompt 模板无关，这部分逻辑不变
        c_focus_ranges, r_focus_ranges = self.get_dual_diff_ranges(chosen, rejected)

        # 3. 注入 Mask - Chosen
        chosen_input_ids = chosen_tokens["input_ids"]
        chosen_labels = [IGNORE_INDEX] * len(chosen_input_ids) # 默认全 Mask
        c_offsets = chosen_tokens["offset_mapping"]
        
        # 计算 prompt 在 chosen tokens 中的长度 (动态计算，适配模板长度)
        prompt_char_len = len(prompt)
        prompt_token_len_c = 0
        for idx, (start, end) in enumerate(c_offsets):
            if start < prompt_char_len:
                prompt_token_len_c = idx + 1
            else:
                # 找到了 Response 的起始点
                break
                
        for idx, (start, end) in enumerate(c_offsets):
            if idx < prompt_token_len_c: continue # 跳过 Prompt (Context)
            
            # 映射到 response (code) 内的相对字符位置
            token_start = start - prompt_char_len
            token_end = end - prompt_char_len
            
            if token_start >= 0:
                is_focus = False
                for (fs, fe) in c_focus_ranges:
                    # 判断是否落在 Focus 区域 (差异点)
                    if max(token_start, fs) < min(token_end, fe):
                        is_focus = True
                        break
                if is_focus:
                    # 只有差异点/修正点才计算 Loss
                    chosen_labels[idx] = chosen_input_ids[idx]

        # 4. 注入 Mask - Rejected
        rejected_input_ids = rejected_tokens["input_ids"]
        rejected_labels = [IGNORE_INDEX] * len(rejected_input_ids) # 默认全 Mask
        r_offsets = rejected_tokens["offset_mapping"]
        
        prompt_token_len_r = 0
        for idx, (start, end) in enumerate(r_offsets):
            if start < prompt_char_len:
                prompt_token_len_r = idx + 1
            else:
                break

        for idx, (start, end) in enumerate(r_offsets):
            if idx < prompt_token_len_r: continue # 跳过 Prompt
            
            token_start = start - prompt_char_len
            token_end = end - prompt_char_len
            
            if token_start >= 0:
                is_focus = False
                for (fs, fe) in r_focus_ranges:
                    # 判断是否落在 Focus 区域 (错误点)
                    if max(token_start, fs) < min(token_end, fe):
                        is_focus = True
                        break
                if is_focus:
                    # 只有错误点才计算 Loss
                    rejected_labels[idx] = rejected_input_ids[idx]

        # 5. 组装返回
        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_input_ids"] = chosen_input_ids
        batch["chosen_attention_mask"] = chosen_tokens["attention_mask"]
        batch["chosen_labels"] = chosen_labels
        batch["rejected_input_ids"] = rejected_input_ids
        batch["rejected_attention_mask"] = rejected_tokens["attention_mask"]
        batch["rejected_labels"] = rejected_labels
        
        return batch

# ===================== 主函数 =====================
def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 加载模型
    print(f"Loading Policy Model from {script_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, use_cache=False)
    
    print(f"Loading Reference Model from {script_args.model_name_or_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, use_cache=False)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    print(f"Loading raw dataset from {script_args.data_path}...")
    dataset = load_dataset('json', data_files=script_args.data_path, split='train')
    
    print("="*40)
    print(f"Sample Prompt Preview (SFT-Aligned):\n{dataset[0]['prompt'][:200]}...")
    print("="*40)

    # 使用双向 Mask Trainer
    trainer = DualMaskedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting Dual-Sided Mask-DPO training...")
    trainer.train()
    
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()