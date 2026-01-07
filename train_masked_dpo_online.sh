#!/bin/bash
# ================= 日志配置 =================
# 定义根目录 (假设当前目录为根目录，或者你可以修改为绝对路径)
ROOT="/data/private/ExeCoder"

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S-%N" # precise time
}

# 创建唯一的日志目录
LOG_PATH=$ROOT/loginfo/dpo_masked_online_$(timestamp)-$RANDOM
mkdir -p "$LOG_PATH"

# 创建一个空的输出文件 (可选，tee 会自动创建)
# > "$LOG_PATH/out.txt" 2>&1 

echo "Logging to: $LOG_PATH/out.txt"

# ================= 训练配置 =================
# 8 卡 L40 并行训练
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 基础模型和数据路径
SFT_MODEL_PATH="/data/private/ExeCoder/results/Qwen2.5-Coder-7B-Instruct-code/checkpoint-327"
DATA_PATH="/data/private/ExeCoder/data/split_dpo_data.json"
OUTPUT_DIR="/data/private/ExeCoder/results/Qwen_dpo_online_mask_v3"

# ================= 启动命令 =================
# 注意末尾的 2>&1 | tee ... 是关键，它将标准输出和错误都同时打印到屏幕和文件
accelerate launch --config_file src/configs/accelerate_config.yaml src/train_online_dpo_mask_v1.py \
    --sft_model_path "$SFT_MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_generations 4 \
    --max_new_tokens 384 \
    --temperature 1.2 \
    --beta 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --learning_rate 5e-7 \
    --num_train_epochs 1 \
    --save_steps 500 \
    2>&1 | tee -a "$LOG_PATH/out.txt"