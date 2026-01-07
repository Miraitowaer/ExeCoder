import os
import subprocess
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


def load_local_model(model_path):
    """加载本地大模型和tokenizer"""
    print(f"从本地路径加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/public/models/base/Qwen/Qwen2.5-Coder-7B-Instruct",
        # model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def format_code_to_single_line(code):
    """
    将Java代码转换为单行格式：
    1. 去除代码块标记（```java和```）
    2. 去除所有换行符
    3. 合并多个连续空格为单个空格
    4. 去除首尾空格
    """
    # 1. 去除代码块标记（如```java和```）
    code = code.replace("```java", "").replace("```", "").strip()
    
    # 2. 去除所有换行符（包括\n和\r）
    code = code.replace("\n", "").replace("\r", "")
    
    # 3. 将多个连续空格（包括制表符）替换为单个空格
    code = re.sub(r'\s+', ' ', code)  # \s+匹配任意空白字符（空格、制表符等）
    
    # 4. 去除首尾多余空格
    code = code.strip()
    
    return code


def generate_translation(model, tokenizer, prompt, src_code, max_new_tokens=2048):
    """调用本地模型生成翻译结果"""
    full_input = f"{prompt}\n{src_code}"
    
    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )
    
    input_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    formatted_text = format_code_to_single_line(generated_text)
    return formatted_text


def parse_evaluation_output(stdout):
    """解析评估脚本输出，提取各项指标"""
    metrics = {
        "compilation_succ": -1,       # 编译成功数
        "bleu": -1.0,                 # BLEU分数
        "exact_match": -1.0,          # 精确匹配率
        "ngram": -1.0,                # Ngram匹配率
        "weighted_ngram": -1.0,       # 加权Ngram匹配率
        "syntax": -1.0,               # 语法匹配率
        "dataflow": -1.0,             # 数据流匹配率
        "codebleu": -1.0,             # CodeBLEU分数
        "runtimeEqCount": -1          # 运行时等价成功数
    }
    
    # 解析编译成功率
    match = re.search('Success - (\d+)', stdout)
    if match:
        metrics["compilation_succ"] = int(match.group(1))
    
    # 解析BLEU和精确匹配
    match = re.search('BLEU:\s*(\d+\.\d+)', stdout)
    if match:
        metrics["bleu"] = float(match.group(1))
    
    match = re.search('Exact Match:\s*(\d+\.\d+)', stdout)
    if match:
        metrics["exact_match"] = float(match.group(1))
    
    # 解析CodeBLEU相关指标
    match = re.search('Ngram match:\s*(\d+\.\d+)', stdout)
    if match:
        metrics["ngram"] = float(match.group(1))
    
    match = re.search('Weighted ngram:\s*(\d+\.\d+)', stdout)
    if match:
        metrics["weighted_ngram"] = float(match.group(1))
    
    match = re.search('Syntax match:\s*(\d+\.\d+)', stdout)
    if match:
        metrics["syntax"] = float(match.group(1))
    
    match = re.search('Dataflow match:\s*(\d+\.\d+)', stdout)
    if match:
        metrics["dataflow"] = float(match.group(1))
    
    match = re.search('CodeBLEU score:\s*(\d+\.\d+)', stdout)
    if match:
        metrics["codebleu"] = float(match.group(1))
    
    # 解析运行时等价性
    match = re.search('Success-RuntimeEq - (\d+)', stdout)
    if match:
        metrics["runtimeEqCount"] = int(match.group(1))
    
    return metrics


def run_evaluation(
    pred_file, 
    ref_file, 
    id_file, 
    dest_lang, 
    eval_scripts_dir,  # 即AVATAR_data目录：/data/public/models/base/Qwen/AVATAR_data
    temp_dir
):
    # 1. 模块所在的根目录是AVATAR_data（因为codegen和evaluation在这里面）
    module_root = eval_scripts_dir  # 直接使用AVATAR_data目录作为模块根目录
    
    # 2. 定义评估命令（与之前相同）
    commands = [
        [
            "python", 
            os.path.join(eval_scripts_dir, "evaluation/compile.py"),
            "--input_file", pred_file,
            "--language", dest_lang,
            "--writeDir", temp_dir
        ],
        [
            "python", 
            os.path.join(eval_scripts_dir, "evaluation/evaluator.py"),
            "--references", ref_file,
            "--txt_ref",
            "--predictions", pred_file,
            "--language", dest_lang
        ],
        [
            "python", 
            os.path.join(eval_scripts_dir, "evaluation/CodeBLEU/calc_code_bleu.py"),
            "--ref", ref_file,
            "--txt_ref",
            "--hyp", pred_file,
            "--lang", dest_lang
        ],
        [
            "python", 
            os.path.join(eval_scripts_dir, "data_LARGE/runtimeEquivalence_corrections/check_runtimeOutput.py"),
            "--input_file", pred_file,
            "--id_file", id_file,
            "--language", dest_lang,
            "--writeDir", temp_dir
        ]
    ]
    
    # 3. 为命令添加PYTHONPATH=AVATAR_data目录（关键修改）
    env_commands = []
    for cmd in commands:
        cmd_str = " ".join(cmd)
        # 让Python在AVATAR_data目录下搜索模块（codegen和evaluation在这里）
        env_cmd = f"PYTHONPATH={module_root} {cmd_str}"  
        env_commands.append(env_cmd)
    
    # 4. 执行命令（显式传递环境变量，确保子进程生效）
    all_commands = "; ".join(env_commands)
    result = subprocess.run(
        all_commands,
        capture_output=True,
        text=True,
        shell=True,
        env=os.environ.copy()  # 继承当前环境变量，叠加PYTHONPATH
    )
    
    # 解析输出
    if result.returncode != 0:
        print(f"评估命令执行失败，错误日志：{result.stderr}")
        return None
    
    return parse_evaluation_output(result.stdout)


def generate_report(metrics, total_samples, output_path):
    """生成评估报表（文本+JSON格式）"""
    # 计算成功率（基于总样本数）
    report = {
        "基本信息": {
            "评估时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "总样本数": total_samples,
        },
        "评估指标": {
            "编译成功率": f"{metrics['compilation_succ']}/{total_samples} ({metrics['compilation_succ']/total_samples*100:.2f}%)" 
            if metrics['compilation_succ'] != -1 else "N/A",
            "BLEU分数": f"{metrics['bleu']:.2f}" if metrics['bleu'] != -1 else "N/A",
            "精确匹配率": f"{metrics['exact_match']:.2f}%" if metrics['exact_match'] != -1 else "N/A",
            "CodeBLEU分数": f"{metrics['codebleu']:.2f}" if metrics['codebleu'] != -1 else "N/A",
            "Ngram匹配率": f"{metrics['ngram']:.2f}%" if metrics['ngram'] != -1 else "N/A",
            "加权Ngram匹配率": f"{metrics['weighted_ngram']:.2f}%" if metrics['weighted_ngram'] != -1 else "N/A",
            "语法匹配率": f"{metrics['syntax']:.2f}%" if metrics['syntax'] != -1 else "N/A",
            "数据流匹配率": f"{metrics['dataflow']:.2f}%" if metrics['dataflow'] != -1 else "N/A",
            "运行时等价成功率": f"{metrics['runtimeEqCount']}/{total_samples} ({metrics['runtimeEqCount']/total_samples*100:.2f}%)"
            if metrics['runtimeEqCount'] != -1 else "N/A"
        }
    }
    
    # 保存为JSON
    json_report_path = f"{output_path}.json"
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 保存为文本（易读格式）
    txt_report_path = f"{output_path}.txt"
    with open(txt_report_path, 'w', encoding='utf-8') as f:
        f.write("===== 代码翻译评估报表 =====\n\n")
        for section, content in report.items():
            f.write(f"【{section}】\n")
            for key, value in content.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"评估报表已保存至：\n- {json_report_path}\n- {txt_report_path}")
    return report


def extract_translations_from_json(json_file, pred_file):
    """从JSON文件中提取翻译结果用于评估"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取翻译结果
    with open(pred_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item['translated_code']}\n")
    
    return pred_file


def translate_and_evaluate(
    model_path,
    src_lang,
    dest_lang,
    code_file,          # 源语言代码文件（如Python）
    ref_file,           # 参考目标语言代码文件（如Java，用于BLEU/CodeBLEU）
    id_file,            # ID文件
    output_dir,
    eval_scripts_dir,   # 评估脚本根目录（如./AVATAR_data）
    max_new_tokens=4096,
    batch_save_size=5,
    do_translation=True,  # 新增参数：是否进行翻译处理
    do_evaluation=True,   # 新增参数：是否进行评估
    start_index=0         # 新增参数：从第几条数据开始处理（0-based）
):
    """完整流程：翻译→保存→多维度评估→生成报表"""
    # 添加时间戳到输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"results_{timestamp}")
    # output_dir = os.path.join(output_dir, "results_20251023_094516")
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取源文件、参考文件和ID文件
    with open(code_file, 'r', encoding='utf-8') as f:
        code_snippets = [line.rstrip('\n') for line in f.readlines()]
    with open(ref_file, 'r', encoding='utf-8') as f:
        ref_snippets = [line.rstrip('\n') for line in f.readlines()]  # 参考目标代码
    with open(id_file, 'r', encoding='utf-8') as f:
        ids = [line.strip() for line in f.readlines()]
    
    # 校验数据长度
    assert len(code_snippets) == len(ref_snippets) == len(ids), \
        f"源文件({len(code_snippets)})、参考文件({len(ref_snippets)})、ID文件({len(ids)})数量不匹配"
    total_samples = len(code_snippets)
    print(f"加载数据完成，共{total_samples}个样本")
    
    # 校验start_index有效性
    if start_index < 0 or start_index >= total_samples:
        raise ValueError(f"start_index({start_index})必须在[0, {total_samples-1}]范围内")
    
    # 定义JSON结果文件
    json_output = os.path.join(output_dir, f"translated_{src_lang}_to_{dest_lang}.json")
    # json_output = "/data/private/ExeCoder/translation_results/results_20251023_134021/washed_translated_python_to_java.json"
    translated_results = []
    
    # 如果不需要翻译，检查是否有现成的结果文件
    if not do_translation:
        if not os.path.exists(json_output):
            raise FileNotFoundError(f"do_translation=False但未找到现有结果文件: {json_output}")
        
        # 加载现有结果
        with open(json_output, 'r', encoding='utf-8') as f:
            translated_results = json.load(f)
        print(f"已加载现有翻译结果，共{len(translated_results)}条数据")
    else:
        # 如果从中间开始处理，先加载已有的结果（如果存在）
        if start_index > 0 and os.path.exists(json_output):
            with open(json_output, 'r', encoding='utf-8') as f:
                translated_results = json.load(f)
            print(f"已加载现有{len(translated_results)}条翻译结果，将从第{start_index}条开始继续处理")
        
        # 加载模型
        model, tokenizer = load_local_model(model_path)
        
        # 提示词模板
        prompt_template = f"Translate the following {src_lang} program into a {dest_lang} program.\nDo not return anything other than the translated code"
        
        # 存储当前批次结果
        batch_results = []
        
        # 从start_index开始处理样本
        total_to_process = total_samples - start_index
        print(f"将从第{start_index}条数据开始处理，共需处理{total_to_process}条数据")
        
        # 逐样本翻译
        for idx in tqdm(range(start_index, total_samples), total=total_to_process, desc="翻译中"):
            code = code_snippets[idx]
            ref_code = ref_snippets[idx]
            code_id = ids[idx]
            
            translated_code = generate_translation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_template,
                src_code=code,
                max_new_tokens=max_new_tokens
            )
            
            result_item = {
                "id": code_id,
                "source_code": code,
                "reference_code": ref_code,  # 真实目标代码（用于评估）
                "translated_code": translated_code
            }
            
            # 如果是继续处理，替换原有位置的数据，否则追加
            if idx < len(translated_results):
                translated_results[idx] = result_item
            else:
                translated_results.append(result_item)
            
            batch_results.append(result_item)
            
            # 批量保存到JSON
            if (idx - start_index + 1) % batch_save_size == 0:
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(translated_results, f, indent=2, ensure_ascii=False)
                
                batch_results = []
                print(f"\n已处理{idx + 1}条数据，保存至{json_output}")
        
        # 处理剩余数据
        if batch_results:
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(translated_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n处理完成，共处理{len(translated_results)}条数据，已保存至{json_output}")
    
    print(f"完整结果JSON已保存至: {json_output}")
    
    # 如果需要评估
    if do_evaluation:
        # 准备评估所需文件
        pred_file = os.path.join(output_dir, "predicted_code_temp.txt")
        # 从JSON文件中提取翻译结果用于评估
        extract_translations_from_json(json_output, pred_file)
        
        id_mapping_file = os.path.join(output_dir, "ids.txt")
        with open(id_mapping_file, 'w', encoding='utf-8') as f:
            for item in translated_results:
                f.write(f"{item['id']}\n")
        
        # 运行多维度评估
        print("开始多维度评估...")
        eval_temp_dir = os.path.join(output_dir, "eval_temp")  # 评估临时文件目录
        metrics = run_evaluation(
            pred_file=pred_file,
            ref_file=ref_file,          # 传入真实目标代码文件作为参考
            id_file=id_mapping_file,
            dest_lang=dest_lang,
            eval_scripts_dir=eval_scripts_dir,
            temp_dir=eval_temp_dir
        )
        
        if not metrics:
            print("评估失败，无法生成报表")
            return
        
        # 生成并保存评估报表
        report_path = os.path.join(output_dir, "evaluation_report")
        generate_report(metrics, total_samples, report_path)
        
        # 清理临时文件
        if os.path.exists(eval_temp_dir):
            import shutil
            shutil.rmtree(eval_temp_dir)
            print("评估临时文件已清理")
        
        if os.path.exists(pred_file):
            os.remove(pred_file)
            print("评估用临时预测文件已清理")
    else:
        print("do_evaluation=False，跳过评估步骤")


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/data/public/models/base/Qwen/Qwen2.5-Coder-7B-Instruct"
    SRC_LANG = "python"
    DEST_LANG = "java"
    CODE_FILE = "/data/private/ExeCoder/AVATAR_data/data_LARGE/test.java-python.python"  # 源语言代码（Python）
    REF_FILE = "/data/private/ExeCoder/AVATAR_data/data_LARGE/test.java-python.java"     # 参考目标代码（Java，用于评估）
    ID_FILE = "/data/private/ExeCoder/AVATAR_data/data_LARGE/test.java-python.id"
    OUTPUT_DIR = "/data/private/ExeCoder/translation_results"
    EVAL_SCRIPTS_DIR = "/data/private/ExeCoder/AVATAR_data"  # 评估脚本根目录（包含evaluation、data_LARGE等）
    MAX_NEW_TOKENS = 2048
    BATCH_SAVE_SIZE = 1
    
    # 新增参数配置
    DO_TRANSLATION = True    # 是否进行翻译处理
    DO_EVALUATION = True     # 是否进行评估
    START_INDEX = 0          # 从第几条数据开始处理（0-based索引）
    
    translate_and_evaluate(
        model_path=MODEL_PATH,
        src_lang=SRC_LANG,
        dest_lang=DEST_LANG,
        code_file=CODE_FILE,
        ref_file=REF_FILE,
        id_file=ID_FILE,
        output_dir=OUTPUT_DIR,
        eval_scripts_dir=EVAL_SCRIPTS_DIR,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_save_size=BATCH_SAVE_SIZE,
        do_translation=DO_TRANSLATION,
        do_evaluation=DO_EVALUATION,
        start_index=START_INDEX
    )