import json
import argparse
import sys

# ANSI 颜色代码，用于终端高亮
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_code_with_highlight(code, error_lines, title="Rejected Code"):
    print(f"{BLUE}--- {title} ---{RESET}")
    lines = code.split('\n')
    
    # 找到最小和最大行号，为了展示方便，只展示错误行附近的上下文
    if error_lines:
        min_line = max(0, min(error_lines) - 5)
        max_line = min(len(lines), max(error_lines) + 5)
    else:
        min_line = 0
        max_line = min(len(lines), 20) # 默认只看前20行

    for i in range(min_line, max_line):
        line_content = lines[i]
        # 注意：error_lines 是 0-indexed
        if i in error_lines:
            # 高亮显示错误行
            print(f"{RED}>> {i+1:4d} | {line_content}{RESET}  <-- MASKED (Weight=1.0)")
        else:
            # 普通行 (Protected)
            print(f"{GREEN}   {i+1:4d} | {line_content}{RESET}")
    
    if max_line < len(lines):
        print(f"   ... ({len(lines) - max_line} more lines) ...")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="./data/dpo_train_all.json", help="Path to the mined json file")
    parser.add_argument('--type', type=str, default="all", choices=['all', 'compilation_error', 'execution_error', 'logic_error'])
    parser.add_argument('--limit', type=int, default=3, help="How many samples to show per type")
    args = parser.parse_args()

    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        return

    print(f"Loaded {len(data)} samples.")
    
    # 按类型分组
    grouped = {'compilation_error': [], 'execution_error': [], 'logic_error': []}
    for item in data:
        etype = item.get('error_type')
        if etype in grouped:
            grouped[etype].append(item)

    # 开始展示
    target_types = [args.type] if args.type != 'all' else grouped.keys()

    for etype in target_types:
        samples = grouped[etype]
        print(f"\n{YELLOW}==========================================")
        print(f"Reviewing: {etype.upper()} (Found: {len(samples)})")
        print(f"=========================================={RESET}")
        
        if not samples:
            print("No samples found for this type.")
            continue

        # 展示前 N 个
        for i, item in enumerate(samples[:args.limit]):
            print(f"\n{YELLOW}Sample #{i+1} / {len(samples)}{RESET}")
            print(f"Prompt Snippet: {item['prompt'][:100].replace(chr(10), ' ')}...")
            print(f"Error Message : {item.get('error_msg', 'N/A')}")
            print(f"Error Lines   : {item.get('error_lines')}")
            
            # 核心：可视化 Rejected 代码
            print_code_with_highlight(item['rejected'], item.get('error_lines', []))
            
            # 交互式确认 (可选)
            # input("Press Enter to see next...") 

if __name__ == "__main__":
    main()