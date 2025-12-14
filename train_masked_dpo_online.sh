#!/bin/bash
# 8 卡 L40 并行训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SFT_MODEL="/data/private/ExeCoder/results/Deepseek-coder-6.7b-instruct-code-sft-aligned"
DATA_PATH="/data/private/ExeCoder/data/train_data_sft_with_tests.json"
OUTPUT_DIR="/data/private/ExeCoder/results/trl_online_mask_dpo"

accelerate launch --config_file src/configs/accelerate_config.yaml src/train_trl_online_mask.py \
    --sft_model_path $SFT_MODEL \
    --data_path $DATA_PATH \
    --lang cpp \
    --num_generations 4 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 50 \
    --bf16 True