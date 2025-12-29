#!/usr/bin/bash

# 显存状态检查
nvidia-smi

ROOT=/data/private/ExeCoder
MODEL_PATH=$ROOT

BASE_DATA_PATH=$ROOT/data/testset
BASE_MODEL_PATH=$ROOT/checkpoint

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S-%N" # current time
}

LOG_PATH=$ROOT/loginfo/$(timestamp)-$RANDOM
mkdir -p "$LOG_PATH"

# 1. 再次确认模型路径 (根据你实际下载的文件夹名修改)
# MODELS=("Deepseek-coder-6.7b-instruct")
MODELS=("CodeLlama-7B-Instruct") 

# 数据模态
MODALs=("code")

# --data_path $MODEL_PATH/data/XLCoST_data/XLCoST-Instruct/Tuning/$MODAL/train.json \

length=${#MODALs[@]}

for (( i=0; i<$length; i++ ))
do
  MODAL=${MODALs[$i]}
  MODEL=${MODELS[$i]}
  
  echo "Starting training for Model: $MODEL with Data: $MODAL on 8xL40"

  # 2. 启动训练
  # 建议加上 --master_port 防止端口冲突
  deepspeed --master_port=$((29500 + $RANDOM % 1000)) $ROOT/src/train.py \
      --model_name_or_path $MODEL_PATH/checkpoint/$MODEL \
      --data_path /data/private/ExeCoder/data/split_sft_data.json \
      --dev_data_path $MODEL_PATH/data/XLCoST_data/XLCoST-Instruct/Tuning/$MODAL/dev.json \
      --cache_dir $MODEL_PATH/data_cache \
      --output_dir $MODEL_PATH/results/$MODEL-$MODAL \
      --num_train_epochs 3 \
      --model_max_length 4096 \
      --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --evaluation_strategy "steps" \
      --eval_steps 100 \
      --save_strategy "steps" \
      --save_steps 100 \
      --save_total_limit 2 \
      --load_best_model_at_end True \
      --learning_rate 2e-5 \
      --warmup_steps 100 \
      --logging_steps 10 \
      --lr_scheduler_type "cosine" \
      --report_to "tensorboard" \
      --gradient_checkpointing True \
      --deepspeed $ROOT/src/configs/deepspeed_config_zero2_raw.json \
      --fp16 True \
      > "$LOG_PATH/out.txt" 2>&1

done