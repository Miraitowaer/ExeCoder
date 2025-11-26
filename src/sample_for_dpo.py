import json
import os
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def main(args):
    # 1. 加载训练数据 (作为 Prompt 来源)
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        examples = json.load(f)
    
    # 2. 初始化模型
    print(f"Loading model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # 根据你的显卡情况自动调整 tensor_parallel_size
    # 如果是 8 卡 H20/L40，这里建议设为 1 (单卡跑推理足够)，或者 8 (跑得更快)
    # 这里默认设为 1，方便你多卡并行跑不同的任务，或者设为 8 跑得飞快
    tensor_parallel_size = args.tensor_parallel_size 

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096, # 保持和训练一致
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        dtype="auto"
    )

    # 3. 构造 Prompts
    # 使用和推理一致的 Chat Template
    prompts = []
    for ex in examples:
        # 构造输入内容，这里保持和 vllm_inference.py 一致的逻辑
        content = ex['instruction'] + '\n' + ex['input']
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt)

    # 4. 设置 DPO 采样参数 (关键！)
    # temperature=0.8: 增加随机性，诱导模型犯错
    # n=5: 每个题目生成 5 个候选
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature, 
        top_p=0.95,
        max_tokens=2048,
    )

    print(f"Starting generation for {len(prompts)} prompts with N={args.num_samples}...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. 结果拆分与保存
    # 为了复用 evaluation.sh，我们将 N 个样本拆分成 N 个独立的 JSON 文件
    # 例如: train_preds_0.json, train_preds_1.json ...
    
    # 初始化 N 个结果列表
    all_candidates = [[] for _ in range(args.num_samples)]

    for i, output in enumerate(outputs):
        original_example = examples[i]
        
        # 遍历该题目的 N 个生成结果
        for sample_idx in range(args.num_samples):
            # 复制原始元数据 (id, ground_truth 等)
            new_entry = original_example.copy()
            
            # 填入生成的代码
            generated_code = output.outputs[sample_idx].text
            new_entry['generated_output'] = generated_code
            
            # 加入对应的列表中
            all_candidates[sample_idx].append(new_entry)

    # 保存文件
    os.makedirs(args.output_dir, exist_ok=True)
    for sample_idx in range(args.num_samples):
        save_path = os.path.join(args.output_dir, f"dpo_sample_{sample_idx}.json")
        with open(save_path, 'w') as f:
            json.dump(all_candidates[sample_idx], f, indent=4)
        print(f"Saved sample batch {sample_idx} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True, help="Path to train.json")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5, help="Number of samples per prompt")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    args = parser.parse_args()
    main(args)