import json
import os
import argparse
import subprocess
import tempfile
import re
from tqdm import tqdm

def check_java(code):
    # 尝试提取类名，如果找不到默认用 Main
    match = re.search(r'public\s+class\s+(\w+)', code)
    class_name = match.group(1) if match else "Main"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, f"{class_name}.java")
        # 移除 package 声明以防报错
        code = re.sub(r'package\s+[\w\.]+;', '', code)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 编译
        cmd = ["javac", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, result.stderr # 编译失败
        return True, ""

def check_cpp(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp') as f:
        f.write(code)
        f.flush()
        # -fsyntax-only 只检查语法不链接
        cmd = ["g++", "-fsyntax-only", f.name]
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

def main(args):
    print(f"Mining compilation errors from {args.sample_file}...")
    with open(args.sample_file, 'r') as f:
        data = json.load(f)
    
    dpo_pairs = []
    lang = args.lang # cpp, java, python

    success_cnt = 0
    fail_cnt = 0

    for item in tqdm(data):
        prompt = item['instruction'] + '\n' + item['input']
        chosen = item['output'] # 训练集的 Ground Truth 肯定是 Chosen
        rejected = item.get('generated_output', '')
        
        # 清洗代码（移除markdown标记）
        if "```" in rejected:
            pattern = r"```(?:\w+)?\n(.*?)```"
            match = re.search(pattern, rejected, re.DOTALL)
            if match:
                rejected = match.group(1)
        
        is_valid = False
        error_msg = ""

        if lang == 'java':
            is_valid, error_msg = check_java(rejected)
        elif lang == 'cpp':
            is_valid, error_msg = check_cpp(rejected)
        elif lang == 'python':
            is_valid, error_msg = check_python(rejected)
        
        if not is_valid:
            fail_cnt += 1
            # 构造 DPO 数据对
            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,     # 好的代码（人工写的）
                "rejected": rejected, # 坏的代码（编译报错的）
                "error_msg": error_msg # (可选) 也可以把报错信息放进 prompt 训练
            })
        else:
            success_cnt += 1

    print(f"Processing done. Found {fail_cnt} compilation errors out of {len(data)} samples.")
    
    # 保存 DPO 数据
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(dpo_pairs, f, indent=4)
    print(f"Saved {len(dpo_pairs)} DPO pairs to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_file', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--lang', type=str, default='java')
    args = parser.parse_args()
    main(args)