import json
import argparse
import sys
import re

# ANSI 颜色代码，用于终端高亮
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def calculate_line_offset(raw_text: str) -> int:
    """与 preprocess_focused_dpo.py 保持一致的行号偏移量计算"""
    if not raw_text:
        return 0
        
    if "```" in raw_text:
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            code_start_index = match.start(1)
            prefix_text = raw_text[:code_start_index]
            return prefix_text.count('\n')
    return 0

def print_code_with_highlight(code, error_lines, title="Code Snippet", force_full=False):
    """
    打印代码并高亮错误行。
    :param force_full: 是否强制显示完整代码 (不折叠)
    """
    print(f"{BLUE}{BOLD}--- {title} ---{RESET}")
    if not code:
        print("(No code content)")
        return

    lines = code.split('\n')
    
    # 视窗逻辑：
    if force_full:
        # 强制全量展示，不计算 min/max
        min_line = 0
        max_line = len(lines)
    elif error_lines:
        # 如果有错误行，聚焦错误行附近 (前5后5)
        min_line = max(0, min(error_lines) - 5)
        max_line = min(len(lines), max(error_lines) + 5)
    else:
        # 默认展示前 30 行
        min_line = 0
        max_line = min(len(lines), 30) 

    # 打印代码行
    for i in range(min_line, max_line):
        line_content = lines[i]
        # error_lines 是 0-indexed
        if error_lines and i in error_lines:
            # 高亮显示错误行
            print(f"{RED}>> {i+1:4d} | {line_content}{RESET}  <-- [ERROR]")
        else:
            # 普通行
            print(f"{GREEN}   {i+1:4d} | {line_content}{RESET}")
    
    # 尾部折叠提示
    if max_line < len(lines):
        remaining = len(lines) - max_line
        print(f"{BLUE}   ... ({remaining} more lines hidden) ...{RESET}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="./data/dpo_errors_pairs.json", help="Path to the mined json file")
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

    # 确定要展示的类型
    target_types = [args.type] if args.type != 'all' else grouped.keys()

    for etype in target_types:
        samples = grouped[etype]
        if not samples: continue

        print(f"\n{YELLOW}{'='*60}")
        print(f" REVIEWING: {etype.upper()} (Total Found: {len(samples)})")
        print(f"{'='*60}{RESET}")
        
        # 展示前 N 个
        for i, item in enumerate(samples[:args.limit]):
            print(f"\n{YELLOW}>>> Sample #{i+1} / {len(samples)} ({etype}){RESET}")
            
            # 打印 Prompt 摘要
            prompt_snippet = item['prompt'][:100].replace('\n', ' ')
            print(f"Prompt: {prompt_snippet}...")

            # 统一展示格式：错误信息 + 错误行 + 带高亮的代码
            print(f"Error Message: {item.get('error_msg', 'N/A')}")
            print(f"Error Lines  : {item.get('error_lines')}")
            
            # 新增：对 logic_error 打印 error_code
            if etype == 'logic_error':
                error_code = item.get('error_code', [])
                print(f"Error Code   : {error_code}")  # 打印错误代码片段
            
            # 展示 rejected 代码
            print_code_with_highlight(
                item.get('rejected', ''), 
                # 计算显示的错误行（相对提取出的代码）
                [line - calculate_line_offset(item.get('rejected', '')) for line in item.get('error_lines', [])], 
                title=f"REJECTED CODE ({etype})",
                force_full=False
            )
            
            # 对执行错误和逻辑错误，额外展示 chosen 代码
            if etype in ['execution_error', 'logic_error']:
                print_code_with_highlight(
                    item.get('chosen', ''), 
                    error_lines=[],  # chosen 代码默认无错误行高亮
                    title=f"CHOSEN CODE ({etype})",
                    force_full=False
                )

if __name__ == "__main__":
    main()