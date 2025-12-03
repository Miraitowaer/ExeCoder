import json
import argparse
import sys

# ANSI 颜色代码，用于终端高亮
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

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
        # [修改点] 强制全量展示，不计算 min/max
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
            # 高亮显示错误行 (红底或者红字)
            print(f"{RED}>> {i+1:4d} | {line_content}{RESET}  <-- [MASKED]")
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

            # =========================================================
            # 针对 Logic Error 的增强展示：Chosen vs Rejected + Trace
            # =========================================================
            if etype == 'logic_error':
                print(f"\n{BOLD}COMPARISON VIEW:{RESET}")
                
                # 1. 展示 Chosen Code (基准) - [修改点] 强制全显示
                print_code_with_highlight(
                    item.get('chosen', ''), 
                    [], 
                    title="CHOSEN CODE (Reference)",
                    force_full=True  # <-- 开启全景模式
                )

                # 2. 展示 Rejected Code (预测) - [修改点] 强制全显示
                print_code_with_highlight(
                    item.get('rejected', ''), 
                    item.get('error_lines', []), 
                    title="REJECTED CODE (Prediction)",
                    force_full=True  # <-- 开启全景模式，不再隐藏行
                )

                # 3. 展示 Trace 表格
                print(f"{CYAN}{BOLD}--- TRACE DIVERGENCE ANALYSIS ---{RESET}")
                print(item.get('error_msg', 'N/A'))
                print(f"{CYAN}{'-'*60}{RESET}")

            # =========================================================
            # 其他错误类型 (Compilation / Execution) - 保持折叠
            # =========================================================
            else:
                print(f"Error Message: {item.get('error_msg', 'N/A')}")
                print(f"Error Lines  : {item.get('error_lines')}")
                
                print_code_with_highlight(
                    item.get('rejected', ''), 
                    item.get('error_lines', []), 
                    title="REJECTED CODE",
                    force_full=False # 编译错误代码可能很长且无关，保持折叠
                )

            # 交互式暂停
            # input(f"\n{BOLD}[Press Enter to continue...]{RESET}")

if __name__ == "__main__":
    main()