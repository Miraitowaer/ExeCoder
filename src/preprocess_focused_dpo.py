import json
import re
import argparse
from tqdm import tqdm

def calculate_line_offset(raw_text):
    """
    计算从原始文本到提取出的代码之间的行号偏移量。
    逻辑必须与 mine_compilation_errors.py 中的 extract_code 保持一致。
    """
    if "```" in raw_text:
        # 匹配 ```语言名(可选) \n 代码 ```
        # 注意：这里我们需要找到代码块内容的起始位置
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            # match.start(1) 是代码内容(group 1)在原始字符串中的起始索引
            code_start_index = match.start(1)
            
            # 截取代码开始前的所有文本
            prefix_text = raw_text[:code_start_index]
            
            # 计算前缀中的换行符数量，这就是偏移量
            # 例如：前缀有 3 个 \n，说明代码从第 3 行（0-indexed）即第 4 行开始
            return prefix_text.count('\n')
            
    return 0

def extract_error_lines(error_msg, lang):
    """
    从编译报错信息中提取出错的行号 (0-indexed)。
    包含自适应过滤机制。
    """
    lines = set()
    if not error_msg:
        return list(lines)

    # --- 1. 全局错误过滤 (Global Error Filtering) ---
    # 如果报错包含这些词，说明是环境或依赖问题，不要惩罚具体的代码行
    global_error_patterns = [
        r"was not declared",           # Cpp: 未声明
        r"did you forget to",          # Cpp: 提示缺少 include
        r"cannot find symbol",         # Java: 找不到符号
        r"package .* does not exist",  # Java: 包不存在
        r"ImportError",                # Python: 导包错
        r"ModuleNotFoundError",        # Python: 模块错
        r"is defined in header",       # Cpp: 明确提示在头文件中
        # r"stray",                    # stray 错误通常是字符问题，可以保留行号进行惩罚
        r"symbol: class",              # Java: 找不到类
    ]

    for pattern in global_error_patterns:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return [] # 命中全局错误，回退到标准 DPO

    # --- 2. 提取相对行号 (Relative Line Extraction) ---
    
    # C++ / G++
    if lang in ['cpp', 'c++']:
        matches = re.findall(r':(\d+):\d+: error:', error_msg)
        for m in matches:
            lines.add(int(m) - 1)

    # Java
    elif lang == 'java':
        matches = re.findall(r':(\d+): error:', error_msg)
        for m in matches:
            lines.add(int(m) - 1)

    # Python
    elif lang == 'python':
        matches = re.findall(r'line\s+(\d+)', error_msg)
        for m in matches:
            lines.add(int(m) - 1)

    return sorted(list(lines))

def detect_lang(instruction):
    instruction = instruction.lower()
    if "to java" in instruction:
        return "java"
    elif "to c++" in instruction or "to cpp" in instruction:
        return "cpp"
    elif "to python" in instruction:
        return "python"
    return "unknown"

def main(args):
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_count = 0
    total_errors_found = 0

    for item in tqdm(data, desc="Processing Error Msg"):
        # 1. 确定语言
        lang = detect_lang(item['prompt']) 
        if lang == "unknown" and 'instruction' in item:
            lang = detect_lang(item['instruction'])
        
        # 2. 提取相对行号 (相对于纯代码块)
        error_msg = item.get('error_msg', '')
        relative_error_lines = extract_error_lines(error_msg, lang)
        
        # 3. 计算偏移量 (Key Fix!)
        offset = calculate_line_offset(item['rejected'])
        
        # 4. 加上偏移量，得到在原始 rejected 字符串中的绝对行号
        absolute_error_lines = [line_idx + offset for line_idx in relative_error_lines]

        # 5. 验证越界 (Optional but recommended)
        rejected_code = item['rejected']
        total_lines = len(rejected_code.split('\n'))
        valid_error_lines = [l for l in absolute_error_lines if l < total_lines]

        # 6. 保存
        item['error_lines'] = valid_error_lines
        
        if valid_error_lines:
            total_errors_found += 1
        processed_count += 1

    print(f"Done! Found focused error lines in {total_errors_found}/{len(data)} samples.")
    print(f"Saving augmented data to {args.output_file}...")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)

# python src/preprocess_focused_dpo.py \
#     --input_file /data/private/ExeCoder/data/dpo_mining/dpo_train_data_final.json \
#     --output_file /data/private/ExeCoder/data/dpo_mining/dpo_train_data_focused.json