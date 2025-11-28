import json
import os
import argparse
import subprocess
import tempfile
import re
from tqdm import tqdm

# 提取代码块的通用函数
def extract_code(text):
    if "```" in text:
        # 匹配 ```语言名(可选) \n 代码 ```
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return text

def check_java(code):
    match = re.search(r'public\s+class\s+(\w+)', code)
    class_name = match.group(1) if match else "Main"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, f"{class_name}.java")
        code = re.sub(r'package\s+[\w\.]+;', '', code)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 增加 -nowarn 忽略警告，只看 Error
        cmd = ["javac", "-nowarn", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, result.stderr
        return True, ""

def check_cpp(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp') as f:
        f.write(code)
        f.flush()
        # 添加常用头文件支持（可选），或者只做语法检查
        cmd = ["g++", "-fsyntax-only", "-w", f.name] # -w 关闭警告
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, result.stderr
        return True, ""

def check_python(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
        f.write(code)
        f.flush()
        cmd = ["python", "-m", "py_compile", f.name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, result.stderr
        return True, ""

def detect_target_lang(instruction):
    """
    从指令中解析目标语言。
    假设指令格式如: "Translate the given code from C++ to Java."
    """
    instruction = instruction.lower()
    if "to java" in instruction:
        return "java"
    elif "to c++" in instruction or "to cpp" in instruction:
        return "cpp"
    elif "to python" in instruction:
        return "python"
    return None

def main(args):
    print(f"Mining compilation errors from {args.sample_file}...")
    if not os.path.exists(args.sample_file):
        print(f"File not found: {args.sample_file}")
        return

    with open(args.sample_file, 'r') as f:
        data = json.load(f)
    
    dpo_pairs = []
    stats = {"java": 0, "cpp": 0, "python": 0, "skipped": 0}
    error_counts = {"java": 0, "cpp": 0, "python": 0}

    for item in tqdm(data):
        prompt = item['instruction'] + '\n' + item['input']
        chosen = item['output']
        rejected = item.get('generated_output', '')
        
        # 1. 自动检测目标语言
        lang = detect_target_lang(item['instruction'])
        
        # 如果强制指定了语言，则只处理该语言的数据，且仅处理检测匹配的数据
        if args.force_lang and lang != args.force_lang:
            continue

        if not lang:
            stats["skipped"] += 1
            continue

        # 2. 清洗代码
        code_to_check = extract_code(rejected)
        
        is_valid = True
        error_msg = ""

        # 3. 根据语言分发检查
        if lang == 'java':
            stats["java"] += 1
            is_valid, error_msg = check_java(code_to_check)
        elif lang == 'cpp':
            stats["cpp"] += 1
            is_valid, error_msg = check_cpp(code_to_check)
        elif lang == 'python':
            stats["python"] += 1
            is_valid, error_msg = check_python(code_to_check)
        
        # 4. 如果编译失败，加入 DPO 数据集
        if not is_valid:
            error_counts[lang] += 1
            dpo_pairs.append({
                "prompt": prompt,       # 也可以直接存 prompt
                "chosen": chosen,
                "rejected": rejected,   # 这里存带 Markdown 的原始生成
                "error_msg": error_msg
            })

    print(f"Processing stats: Processed {stats}, Skipped {stats['skipped']}")
    print(f"Errors found: {error_counts}")
    print(f"Total DPO pairs mined: {len(dpo_pairs)}")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(dpo_pairs, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_file', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    # force_lang 可选，如果不传则自动处理所有语言
    parser.add_argument('--force_lang', type=str, default=None, help="Only process specific language (java, cpp, python)")
    args = parser.parse_args()
    main(args)