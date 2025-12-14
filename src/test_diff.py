import difflib

def get_diff_ranges(correct_code, incorrect_code):
    """
    比较正确代码和错误代码，返回 incorrect_code 中需要被 '重点关注' 的行号范围。
    即：找出 incorrect_code 中与 correct_code 不同的部分。
    """
    # 1. 按行分割
    correct_lines = correct_code.splitlines(keepends=True)
    incorrect_lines = incorrect_code.splitlines(keepends=True)
    
    # 2. 创建 SequenceMatcher
    matcher = difflib.SequenceMatcher(None, correct_lines, incorrect_lines)
    
    # correct_lines (a) -> incorrect_lines (b)
    # opcodes 返回五元组: (tag, i1, i2, j1, j2)
    # tag: 'replace', 'delete', 'insert', 'equal'
    # i1, i2: correct_lines 的起止索引
    # j1, j2: incorrect_lines 的起止索引
    
    focus_indices_in_rejected = [] # 存储 rejected 中需要计算 Loss 的行索引
    common_indices_in_rejected = [] # 存储 rejected 中应该被 Mask 掉的行索引

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 这一段是相同的，理论上应该 Mask 掉 (或者权重给低)
            for r_idx in range(j1, j2):
                common_indices_in_rejected.append(r_idx)
        else:
            # 'replace': Rejected 写错了
            # 'insert': Rejected 多写了
            # 这一段是 Rejected 特有的错误，必须保留 Loss (权重给高)
            for r_idx in range(j1, j2):
                focus_indices_in_rejected.append(r_idx)
                
    return set(focus_indices_in_rejected), set(common_indices_in_rejected)

# --- 测试一下 ---
chosen_str = """def add(a, b):
    return a + b
"""
rejected_str = """def add(a, b):
    print("hello")
    return a - b
"""

focus, common = get_diff_ranges(chosen_str, rejected_str)
print("Rejected 中需要惩罚的行索引:", focus) 
# 预期输出: {1, 2} (即 print... 和 return a-b)
print("Rejected 中相同的行索引 (Mask):", common) 
# 预期输出: {0} (即 def add...)