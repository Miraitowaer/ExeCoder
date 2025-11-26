#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 8卡走起

# 你的 SFT 模型路径 (作为起点)
SFT_MODEL_PATH="/data/private/ExeCoder/results/Qwen2.5-Coder-7B-Instruct-code/checkpoint-best"
DATA_PATH="/data/private/ExeCoder/data/dpo_data.json" # 你构建好的数据

deepspeed --master_port=29501 src/train_dpo.py \
    --model_name_or_path $SFT_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir "/data/private/ExeCoder/results/dpo_final" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --logging_steps 10 \
    --save_steps 100 \
    --deepspeed src/configs/deepspeed_config_zero2.json \
    --fp16 True \
    --report_to "tensorboard"