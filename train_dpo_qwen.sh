#!/bin/bash
# scripts/run_native.sh

ROOT="/data/private/ExeCoder"
SFT_MODEL_PATH="/data/private/ExeCoder/results/Qwen2.5-Coder-7B-Instruct-code/checkpoint-327"
DATA_PATH="/data/private/ExeCoder/data/dpo_train_qwen_clean.json"
OUTPUT_DIR="$ROOT/results/dpo_qwen_native"

# 你的 ZeRO-3 配置文件路径
DS_CONFIG="$ROOT/src/configs/deepspeed_config_zero3.json"

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S-%N" # current time
}

LOG_PATH=$ROOT/loginfo/dpo_$(timestamp)-$RANDOM
mkdir -p "$LOG_PATH"

# 使用 deepspeed 启动器
deepspeed --master_port=29508 src/train_dpo_qwen.py \
    --model_name_or_path "$SFT_MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed_config "$DS_CONFIG" \
    --max_length 2048 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --num_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    > "$LOG_PATH/out.txt" 2>&1

PID=$!
echo "Native DPO Training started. PID: $PID"
echo "Log: tail -f $ROOT/loginfo/native_dpo.log"