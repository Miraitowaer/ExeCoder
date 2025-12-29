#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# 显存检查
nvidia-smi

# ================= 配置区 =================
ROOT="/data/private/ExeCoder"
# SFT 模型路径 (请确认这是你最新的 SFT checkpoint)
SFT_MODEL_PATH="/data/private/ExeCoder/results/Qwen2.5-Coder-7B-Instruct-code/checkpoint-327"

# 数据路径 (请确认这是 v5 版本的 ranked 数据)
DATA_PATH="$ROOT/data/merged_result.json"

# 输出目录 (区分于 Baseline)
OUTPUT_DIR="$ROOT/results/qwen_dpo_merged_result"
# =========================================

# 创建日志目录
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

LOG_PATH=$ROOT/loginfo/dpo_masked_$(timestamp)
mkdir -p "$LOG_PATH"

echo "=================================================="
echo "Starting Mask-DPO Training (Focus on Logic Errors)"
echo "Base Model: $SFT_MODEL_PATH"
echo "Dataset:    $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "=================================================="

# 启动训练
# 核心参数说明:
# --per_device_train_batch_size 2: 保持显存安全
# --gradient_accumulation_steps 8: 稍微调大一点，增加 global batch size，让梯度更稳
# --learning_rate 5e-7: DPO 建议低 LR
# --beta 0.1: DPO 温度系数
deepspeed --master_port=29505 src/train_dpo_masked_v2.py \
    --model_name_or_path $SFT_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --save_steps 50 \
    --beta 0.1 \
    --deepspeed src/configs/deepspeed_config_zero3.json \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --remove_unused_columns False \
    > "$LOG_PATH/out.txt" 2>&1

echo "Done. Log saved to $LOG_PATH/out.txt"