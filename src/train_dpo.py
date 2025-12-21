import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig

# ================= 配置参数 =================
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the SFT model"})
    data_path: str = field(metadata={"help": "Path to the DPO dataset (json)"})
    ignore_index: int = field(default=-100, metadata={"help": "Label value to ignore for loss"})

# ================= 核心：模板对齐函数 =================
def apply_deepseek_template(example):
    """
    [关键修改]
    将原始 DPO 数据集中的 Prompt 包装成与 SFT 完全一致的 Alpaca 格式（包含 Preamble）。
    """
    prompt_raw = example["prompt"]
    
    # SFT 中使用的 Preamble (开场白)
    # 必须和你 train.py 中生效的那个版本一模一样，一个标点都不能差！
    ALPACA_PREAMBLE = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    
    # 检查是否已经包含了模板 (防止重复添加)
    if "### Response:" not in prompt_raw:
        # 构造标准的 Alpaca 格式：Preamble + Instruction + Response
        # 注意：这里我们将 prompt_raw (包含指令和代码) 整体放入 ### Instruction: 块中
        # 这是因为我们的 mining 脚本没有区分 instruction 和 input 字段，这是目前兼容性最好的做法
        new_prompt = f"{ALPACA_PREAMBLE}### Instruction:\n{prompt_raw}\n\n### Response:\n"
    else:
        new_prompt = prompt_raw
        
    example["prompt"] = new_prompt
    return example

def main():
    # 1. 解析参数
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 加载 Policy Model (SFT后的模型)
    print(f"Loading Policy Model from {script_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
    )
    
    # 3. 加载 Reference Model (参考模型)
    # 在 DeepSpeed ZeRO-3 模式下，显式加载通常更稳健
    print(f"Loading Reference Model from {script_args.model_name_or_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
    )

    # 4. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
            tokenizer.pad_token_id = 151643 # Qwen 的默认 pad id
            
    # [重要] Qwen2.5 DPO 建议：明确打印一下 ID 确保加载正确
    print(f"🔥 Tokenizer Loaded | EOS: {tokenizer.eos_token_id} | PAD: {tokenizer.pad_token_id}")
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if ref_model is not None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
    print(f"🔧 Model Config Updated: model.config.pad_token_id = {model.config.pad_token_id}")
        
    # 5. 加载数据
    print(f"Loading dataset from {script_args.data_path}...")
    dataset = load_dataset('json', data_files=script_args.data_path, split='train')

    # ================== 核心修改区域 ==================
    print("Applying DeepSeek-Coder (Alpaca) template to prompts...")
    # 对数据集的每一行应用模板转换
    dataset = dataset.map(apply_deepseek_template)
    
    # 打印一条样本进行人工核对 (Sanity Check)
    # 请在日志中检查是否出现了 "### Instruction:"
    print("="*40)
    print(f"Sample Prompt Preview (Aligned with SFT):\n{dataset[0]['prompt'][:200]}...")
    print("="*40)
    # =================================================

    # 6. 初始化 DPOTrainer
    # 使用 trl 库的标准 DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
    )

    # 7. 开始训练
    print("Starting Standard DPO Baseline training (Template-Aligned)...")
    trainer.train()
    
    # 8. 保存模型
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    # 同时保存 tokenizer，方便后续直接加载推理
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()