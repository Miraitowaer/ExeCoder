import json
import difflib
import os
import argparse
import textwrap

# ===================== 颜色配置 =====================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # 字体颜色
    GRAY = "\033[90m"     # 灰色 (被Mask的部分)
    GREEN = "\033[92m"    # 绿色 (Chosen独有)
    RED = "\033[91m"      # 红色 (Rejected独有)
    YELLOW = "\033[93m"   # 黄色 (用于 Msg)
    BLUE = "\033[94m"     # 蓝色 (用于统计信息)
    
    # 背景颜色 (用于强调 Rejected 的错误行)
    BG_RED = "\033[41m\033[37m"   # 红底白字 (Rejected Focus)
    BG_GREEN = "\033[42m\033[37m" # 绿底白字 (Chosen Focus)

def print_header(text):
    print(f"\n{Colors.BOLD}{'='*30} {text} {'='*30}{Colors.RESET}")

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"{Colors.RED}Error: File {filepath} not found.{Colors.RESET}")
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def truncate(text, width):
    """截断过长的代码行，防止破坏对齐"""
    if len(text) > width:
        return text[:width-3] + "..."
    return text.ljust(width)

def get_side_by_side_view(chosen_code: str, rejected_code: str, width=60):
    """
    生成左右并排的对比视图 (含缺失惩罚逻辑)
    """
    chosen_lines = chosen_code.splitlines()
    rejected_lines = rejected_code.splitlines()
    
    matcher = difflib.SequenceMatcher(None, chosen_lines, rejected_lines)
    
    output_lines = []
    
    # 表头
    header_left = "CHOSEN (Reference)".center(width)
    header_right = "REJECTED (Mask Preview)".center(width)
    output_lines.append(f"{Colors.BOLD}{header_left} | {header_right}{Colors.RESET}")
    output_lines.append("-" * (width * 2 + 3))

    mask_count = 0
    total_rej_lines = 0

    # 预处理 opcodes，合并连续的操作以便处理上下文
    opcodes = matcher.get_opcodes()

    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        c_chunk = chosen_lines[i1:i2]
        r_chunk = rejected_lines[j1:j2]
        
        # ---------------- 关键修改开始 ----------------
        # 检查是否是"Delete"操作 (Rejected 漏写了)
        # 如果是 Delete，我们要惩罚 Rejected 中紧接着出现的下一行(如果有的话)
        is_focus_due_to_deletion = False
        if tag == 'delete':
            # 我们无法在当前 chunk 标记 Rejected (因为它是空的)
            # 但我们需要标记 Rejected 序列中 *当前位置* 的后续内容为 Focus
            # 这是一个概念上的标记，实际可视化时，我们会在下一次迭代的 'equal' 中处理
            # 或者，更简单的策略：我们把 Delete 视为一种特殊的 Replace，
            # 这里的 right_view 虽然是空，但在训练代码中，我们需要把 j1 位置的 Token 权重调高。
            
            # 可视化策略：在 Delete 发生的位置，如果右边是空的，
            # 我们查看 opcodes 的下一个动作。
            if idx + 1 < len(opcodes):
                next_tag, _, _, next_j1, next_j2 = opcodes[idx+1]
                # 如果下一个动作是 Equal，说明模型跳过了中间一段直接写了后面的。
                # 我们必须把那个"后面的第一行"标红！
                if next_tag == 'equal':
                    # 这是一个标记位，告诉下一个循环：即使是 Equal，也要标红第一行
                    pass 
        # ---------------- 关键修改结束 ----------------
        
        # 为了可视化简单，我们采用"只要 tag 不是 equal，就标红"的策略
        # 但针对 delete，右边是空的。
        # 我们需要在训练数据构建时，如果检测到 delete，将 rejected_mask[j1] 设为 1
        
        # 补齐长度
        max_len = max(len(c_chunk), len(r_chunk))
        c_chunk += [""] * (max_len - len(c_chunk))
        r_chunk += [""] * (max_len - len(r_chunk))
        
        for k, (c_line, r_line) in enumerate(zip(c_chunk, r_chunk)):
            left_str = truncate(c_line, width)
            right_str = truncate(r_line, width)
            
            left_view = left_str
            right_view = right_str

            # 左侧显示
            if tag == 'replace' or tag == 'delete':
                left_view = f"{Colors.GREEN}{left_str}{Colors.RESET}"
            
            # 右侧显示逻辑 (Mask 核心)
            if tag == 'equal':
                # === 重点修改：缺失惩罚 ===
                # 如果前一个操作是 delete (i1>0 且 chosen[i1-1] 是被删掉的)，
                # 并且这是 equal 块的第一行，那么这一行必须被惩罚！
                # (这里为了脚本简洁，我们很难拿到前一个 opcode，
                # 但在训练脚本中我们会用 index 严谨控制)
                
                # 简单可视化：如果是 equal，先默认灰色
                if r_line:
                    # 临时逻辑：如果这是 equal，但它承接在一个 delete 之后，应该标红。
                    # 在这个脚本里很难完美回溯。
                    # 但我们可以看：如果 Chosen 是 Node (Delete)，Rejected 接下来是 def (Equal)
                    # 这一行 Def 实际上就是"错位"的。
                    
                    right_view = f"{Colors.GRAY}{right_str}{Colors.RESET}" 
                    mask_count += 1
                    total_rej_lines += 1
                else:
                    right_view = right_str 
                    
            elif tag == 'replace' or tag == 'insert':
                if r_line:
                    right_view = f"{Colors.BG_RED}{right_str}{Colors.RESET}"
                    total_rej_lines += 1
                else:
                    right_view = right_str

            elif tag == 'delete':
                 # 右侧为空，无法标红
                 right_view = " " * width
                 
                 # === 训练逻辑预演 ===
                 # 在这里，训练代码会记录：索引 j1 (Rejected的当前位置) 需要 Loss=1
                 # 所以我们在可视化里，可以在右侧打印一个提示
                 right_view = f"{Colors.BG_RED}[MISSING BLOCK PUNISHMENT]{Colors.RESET}".ljust(width)

            output_lines.append(f"{left_view} | {right_view}")
            
    mask_ratio = (mask_count / total_rej_lines * 100) if total_rej_lines > 0 else 0
    return output_lines, mask_ratio

def visualize_pair(idx, entry):
    print_header(f"Sample #{idx} | Task ID: {entry.get('task_id', 'N/A')}")
    
    print(f"Error Type: {Colors.RED}{entry.get('rejected_error_type', 'N/A')}{Colors.RESET}")
    print(f"Source:     {Colors.BOLD}{entry.get('chosen_source', 'N/A')}{Colors.RESET}")
    
    # ================= 修改点：完整展示 Logic Msg =================
    print(f"{Colors.BOLD}Logic Msg:{Colors.RESET}")
    msg = entry.get('rejected_error_msg', '')
    if msg:
        print(f"{Colors.YELLOW}{msg}{Colors.RESET}")
    else:
        print(f"{Colors.GRAY}(No error message){Colors.RESET}")
    # ==========================================================
    
    lines, ratio = get_side_by_side_view(entry['chosen'], entry['rejected'], width=70)
    
    print(f"\nMask Ratio: {Colors.BLUE}{ratio:.1f}% (Gray lines are masked){Colors.RESET}\n")
    
    for line in lines:
        print(line)

def main():
    # 默认文件路径
    INPUT_FILE = "/data/private/ExeCoder/data/dpo_pairs_ranked_v4.json" 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=INPUT_FILE, help="Path to DPO json file")
    parser.add_argument("--n", type=int, default=6194, help="Number of samples to inspect")
    parser.add_argument("--shuffle", action="store_true", help="Randomly shuffle samples")
    args = parser.parse_args()
    
    data = load_data(args.file)
    if not data: return

    print(f"Loaded {len(data)} pairs. Visualizing {args.n} samples...")
    
    if args.shuffle:
        import random
        random.shuffle(data)
    
    for i, entry in enumerate(data[:args.n]):
        visualize_pair(i+1, entry)
        if i < args.n - 1:
            input(f"\n[Press Enter for next sample...]")

if __name__ == "__main__":
    main()