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

# ===================== 自定义 Masked DPO Trainer =====================
class MaskedDPOTrainer(DPOTrainer):
    """
    继承 DPOTrainer，重写 tokenize_row 以注入 Fine-grained Mask 逻辑
    """
    
    def get_diff_ranges(self, chosen_code: str, rejected_code: str) -> List[Tuple[int, int]]:
        """计算字符级的 Diff 范围"""
        chosen_lines = chosen_code.splitlines(keepends=True)
        rejected_lines = rejected_code.splitlines(keepends=True)
        
        rejected_line_offsets = []
        current_pos = 0
        for line in rejected_lines:
            rejected_line_offsets.append((current_pos, current_pos + len(line)))
            current_pos += len(line)

        matcher = difflib.SequenceMatcher(None, chosen_lines, rejected_lines)
        focus_char_ranges = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            elif tag in ('replace', 'insert'):
                if j1 < len(rejected_line_offsets):
                    start_char = rejected_line_offsets[j1][0]
                    end_line_idx = min(j2 - 1, len(rejected_line_offsets) - 1)
                    end_char = rejected_line_offsets[end_line_idx][1]
                    focus_char_ranges.append((start_char, end_char))
            elif tag == 'delete':
                # 错位惩罚：缺失块后的第一行
                if j1 < len(rejected_line_offsets):
                    start_char = rejected_line_offsets[j1][0]
                    end_char = rejected_line_offsets[j1][1]
                    focus_char_ranges.append((start_char, end_char))

        return focus_char_ranges

    def tokenize_row(self, feature: Dict[str, Any], model: Optional = None) -> Dict[str, Any]:
        """
        重写核心处理逻辑：
        1. [修正] 移除模板注入，保持与 SFT 一致的 Raw Input。
        2. 执行 Tokenization。
        3. 计算 Diff Mask 并注入 rejected_labels。
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        
        # [关键修正]：保持 SFT 的"裸跑"风格，不加 Alpaca 模板
        # 因为你的 SFT 数据就是 {instruction}\n{input}，这里的 prompt 也是这个格式。
        # 直接拼接即可。
        
        # 2. Tokenize Chosen
        chosen_tokens = self.tokenizer(prompt + chosen, truncation=True, max_length=self.max_length, padding=False)
        prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding=False)
        
        prompt_len = len(prompt_tokens["input_ids"])
        chosen_labels = chosen_tokens["input_ids"][:]
        
        # Mask 掉 Prompt 部分
        for i in range(min(len(chosen_labels), prompt_len)):
            # [修正] 使用全局常量 IGNORE_INDEX，避免 self.args 报错
            chosen_labels[i] = IGNORE_INDEX

        # 3. Tokenize Rejected (开启 offset_mapping)
        rejected_full_text = prompt + rejected
        rejected_tokens = self.tokenizer(
            rejected_full_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding=False, 
            return_offsets_mapping=True 
        )
        
        rejected_input_ids = rejected_tokens["input_ids"]
        # [修正] 默认全 Mask (Loss=0)
        rejected_labels = [IGNORE_INDEX] * len(rejected_input_ids) 
        offsets = rejected_tokens["offset_mapping"]
        
        # 4. 应用 Mask 逻辑
        focus_ranges = self.get_diff_ranges(chosen, rejected)
        prompt_char_len = len(prompt)
        
        for idx, (start, end) in enumerate(offsets):
            # 跳过 Prompt 部分
            if idx < prompt_len:
                continue
            
            # 映射到 response (code) 内的相对字符位置
            token_start = start - prompt_char_len
            token_end = end - prompt_char_len
            
            if token_start >= 0:
                is_focus = False
                for (fs, fe) in focus_ranges:
                    # 只要 Token 与 Diff 区域有交集，就保留 Loss
                    if max(token_start, fs) < min(token_end, fe):
                        is_focus = True
                        break
                
                if is_focus:
                    # 只有错误的/差异的部分，才计算 Loss
                    rejected_labels[idx] = rejected_input_ids[idx]
        
        # 5. 组装返回
        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        
        batch["chosen_input_ids"] = chosen_tokens["input_ids"]
        batch["chosen_attention_mask"] = chosen_tokens["attention_mask"]
        batch["chosen_labels"] = chosen_labels
        
        batch["rejected_input_ids"] = rejected_input_ids
        batch["rejected_attention_mask"] = rejected_tokens["attention_mask"]
        batch["rejected_labels"] = rejected_labels # <--- Masked Labels
        
        return batch

# ===================== 主函数 =====================
def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 1. 加载模型
    print(f"Loading Policy Model from {script_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
    )
    
    # 2. 加载参考模型 (ZeRO-3 兼容)
    print(f"Loading Reference Model from {script_args.model_name_or_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
    )

    # 3. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 加载原始数据
    print(f"Loading raw dataset from {script_args.data_path}...")
    dataset = load_dataset('json', data_files=script_args.data_path, split='train')
    
    # 打印预览，确认 Prompt 是 SFT 风格的 Raw Text
    print("="*40)
    print(f"Sample Prompt Preview (SFT-Aligned):\n{dataset[0]['prompt'][:200]}...")
    print("="*40)

    # 5. 使用自定义 Masked Trainer
    trainer = MaskedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 6. 开始训练
    print("Starting Mask-DPO training...")
    trainer.train()
    
    # 7. 保存结果
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()