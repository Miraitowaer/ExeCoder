#!/bin/bash

# 1. 配置路径 (请根据你的实际情况修改)
ROOT=/data/private/ExeCoder
MODEL_PATH="$ROOT/results/Deepseek-coder-6.7b-instruct-code/checkpoint-400" # 你的 SFT 模型
TRAIN_DATA="$ROOT/data/XLCoST_data/XLCoST-Instruct/Tuning/code/train.json" # 训练集
OUTPUT_DIR="$ROOT/data/dpo_mining" # 存放采样结果的地方

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S-%N" # current time
}
LOG_PATH=$ROOT/loginfo/mine_$(timestamp)-$RANDOM
mkdir -p "$LOG_PATH"

# 显卡设置
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Step 1: Generating Samples..."
# 这里的 tensor_parallel_size=8 表示用 8 卡跑一个模型，速度极快
python src/sample_for_dpo.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --num_samples 5 \
    --temperature 0.8 \
    --tensor_parallel_size 8 \
    > "$LOG_PATH/generating_samples.txt" 2>&1

echo "Step 2: Evaluating Samples to find failures..."

# 循环评估生成的 5 个文件
for i in {0..4}
do
    SAMPLE_FILE="dpo_sample_${i}.json"
    echo "Evaluating Batch $i: $SAMPLE_FILE"
    
    # 复用你已经修好的 evaluation 脚本逻辑
    # 注意：这里我们只跑 C++ -> Python/Java 等特定任务，或者你可以跑全量
    # 下面以 C++ -> Java 为例 (因为 Java 报错信息最丰富，适合 DPO)
    
    # 1. 清理旧的生成代码 (防止混淆)
    rm -rf evaluation/evaluator/execution_based/transcoder_evaluation_code/DPO_Mining_Batch_${i}
    
    # 2. 运行评估 (注意 --model 参数只是为了生成文件夹名，我们可以自定义)
    # 我们利用 transcoder_eval.py 会生成 _ordered_unsuccessful.txt 的特性
    python evaluation/evaluator/execution_based/transcoder_eval.py \
        --source_lang "cpp" \
        --target_lang "java" \
        --eval_file "$OUTPUT_DIR/$SAMPLE_FILE" \
        --model "DPO_Mining_Batch_${i}" \
        > "$LOG_PATH/evaluating_batch_${i}.txt" 2>&1
    
    echo "Batch $i Evaluation Done. Check reports in evaluation/evaluator/execution_based/transcoder_evaluation_reports/DPO_Mining_Batch_${i}"
done

echo "All Done! Ready to construct DPO dataset."