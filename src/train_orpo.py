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
# 1. 修改导入：从 trl 导入 ORPO 相关的类
from trl import ORPOTrainer, ORPOConfig

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the SFT model"})
    data_path: str = field(metadata={"help": "Path to the DPO/ORPO dataset (json)"})

def main():
    # 2. 解析参数 (使用 ORPOConfig)
    parser = HfArgumentParser((ScriptArguments, ORPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 3. 加载 Model
    print(f"Loading Model from {script_args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False,
        # 注意：在 DeepSpeed 环境下，不要手动指定 device_map="auto"，交给 DeepSpeed 接管
    )
    
    # [关键修改] ORPO 不需要 Reference Model，这里删除了 ref_model 的加载代码
    # 这将显著降低显存占用

    # 4. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. 加载数据
    # ORPO 需要的数据格式与 DPO 完全一致 (prompt, chosen, rejected)
    dataset = load_dataset('json', data_files=script_args.data_path, split='train')

    # 6. 初始化 ORPOTrainer
    trainer = ORPOTrainer(
        model=model,
        # ref_model=None, # ORPO 不需要这个参数
        args=training_args,  # 这里传入的是 ORPOConfig
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 7. 开始训练
    print("Starting ORPO training...")
    trainer.train()
    
    # 8. 保存模型
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    # 同时保存 tokenizer 方便后续使用
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()