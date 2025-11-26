import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from trl import DPOTrainer

# 定义参数类
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the SFT model"})
    data_path: str = field(metadata={"help": "Path to the DPO dataset (json)"})
    beta: float = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
    max_length: int = field(default=2048, metadata={"help": "Max length"})
    max_prompt_length: int = field(default=1024, metadata={"help": "Max prompt length"})

def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 1. 加载 SFT 后的模型 (作为 Policy Model 和 Reference Model)
    # DPO 需要两个模型，但 DPOTrainer 会自动帮你把 model 复制一份作为 ref_model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False
    )
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载数据
    # 数据集格式必须包含三列：['prompt', 'chosen', 'rejected']
    dataset = load_dataset('json', data_files=script_args.data_path, split='train')

    # 4. 初始化 DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # 设置为 None，Trainer 会自动加载一份 model 的副本作为 ref_model
        args=training_args,
        beta=script_args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
    )

    # 5. 开始训练
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()