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

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the SFT model"})
    data_path: str = field(metadata={"help": "Path to the DPO dataset (json)"})

def main():
    # 1. 解析参数
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 加载 Policy Model (当前训练的模型)
    print(f"Loading Policy Model from {script_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
        # 注意：在 DeepSpeed 环境下，不要手动指定 device_map="auto"，交给 DeepSpeed 接管
    )
    
    # 3. 加载 Reference Model (显式加载以兼容 ZeRO-3)
    # DPO 需要一个参考模型来计算 KL 散度，防止跑偏
    print(f"Loading Reference Model from {script_args.model_name_or_path}...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
    )

    # 4. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. 加载数据
    dataset = load_dataset('json', data_files=script_args.data_path, split='train')

    # 6. 初始化 DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model, # <--- 关键修改：手动传入 ref_model
        args=training_args,  # DPOConfig
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 7. 开始训练
    print("Starting training...")
    trainer.train()
    
    # 8. 保存模型
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    # 同时保存 tokenizer 方便后续使用
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()