from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
from pathlib import Path
import torch


def generate_batch(examples, tokenizer, llm, model: str):
    stop = None
    
    # [关键修改] 定义与 SFT train.py 完全一致的 Alpaca 模板
    # 必须包含 Preamble (Below is an instruction...)
    # ALPACA_PROMPT_DICT = {
    #     "prompt_input": (
    #         "Below is an instruction that describes a task, paired with an input that provides further context. "
    #         "Write a response that appropriately completes the request.\n\n"
    #         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    #     ),
    #     "prompt_no_input": (
    #         "Below is an instruction that describes a task. "
    #         "Write a response that appropriately completes the request.\n\n"
    #         "### Instruction:\n{instruction}\n\n### Response:"
    #     ),
    # }
    
    DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant. Provide only the correct code solution."

    PROMPT_DICT = {
        # 场景 A: 包含 instruction (题目) 和 input (具体输入/上下文)
        "prompt_input": (
            "<s>[INST] <<SYS>>\n"
            "{system_prompt}\n"
            "<</SYS>>\n\n"
            "{instruction}\n\n"
            "{input} [/INST]"
        ),
        
        # 场景 B: 只包含 instruction (题目)
        "prompt_no_input": (
            "<s>[INST] <<SYS>>\n"
            "{system_prompt}\n"
            "<</SYS>>\n\n"
            "{instruction} [/INST]"
        ),
    }

    prompts = []
    for ex in examples:
        # 提取字段，做简单的防御性处理
        instruction = ex.get('instruction', '')
        input_data = ex.get('input', '')

        # 逻辑判断：如果 input 字段存在且不为空，使用 prompt_input 模板
        if input_data and input_data.strip():
            prompt = PROMPT_DICT["prompt_input"].format(
                instruction=instruction,
                input=input_data,
                system_prompt=DEFAULT_SYSTEM_PROMPT
            )
        else:
            # 否则使用 prompt_no_input 模板
            prompt = PROMPT_DICT["prompt_no_input"].format(
                instruction=instruction,
                system_prompt=DEFAULT_SYSTEM_PROMPT
            )
        
        prompts.append(prompt)

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=stop
    )

    # 打印一条 Prompt 确认格式是否正确 (包含 Preamble 和 ### Input:)
    print("="*40)
    print("Inference Prompt Preview:\n{}".format(prompts[0][:500]))
    print("="*40)

    outputs = llm.generate(prompts, sampling_params)
    for i in range(len(examples)):
        examples[i]['generated_output'] = outputs[i].outputs[0].text

    return examples

def generate_main(data_path: str, model_name_or_path: str, saved_path: str, cot: bool=False):
    examples = json.load(open(data_path, 'r'))
    def _convert_for_sft(ex):
        ex['instruction'] = ex["instruction"] + "\nYou need first write a step-by-step outline and then write the code."
        return ex
    
    if cot:
        examples = [_convert_for_sft(x) for x in examples]
        saved_path = saved_path.replace(".jsonl", ".cot.jsonl")

    print(model_name_or_path)
    print("Model `{}`, COT = {}:{}".format(model_name_or_path, cot, model_name_or_path))
    print("Saved path: {}".format(saved_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
    
    num_gpus = torch.cuda.device_count()

    # Create an LLM.
    llm = LLM(
        model=model_name_or_path,
        pipeline_parallel_size=1,
        tensor_parallel_size=num_gpus,
        max_num_seqs=512,
        max_num_batched_tokens=8192,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="auto"
    )
    
    generated_examples = generate_batch(examples, tokenizer, llm, model_name_or_path)    
    print("Generate all over!!!")
    with open(saved_path, 'w', encoding='utf-8') as fw:
        json.dump(generated_examples, fw, indent=4)
        # for ex in generated_examples:
        #     fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=Path(__file__).parent.joinpath(f"data/intput.json").as_posix())
    parser.add_argument('--model_name_or_path', type=str, default='deepseek-ai/deepseek-coder-7b-instruct')
    parser.add_argument('--saved_path', type=str, default=f'output/output.json')
    parser.add_argument('--cot', action='store_true', default=False)
    args = parser.parse_args()

    generate_main(
        data_path=args.data_path,
        model_name_or_path=args.model_name_or_path,
        saved_path=args.saved_path,
        cot=args.cot,
    )
