import sys
import inspect
import textwrap

# =============================================================================
# 1. 待测试的代码样本 (模拟 DPO 数据)
# =============================================================================

# Chosen: 正确的逻辑 (斐波那契数列，累加)
# 3行代码完成逻辑
code_chosen = """
def solution(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""

# Rejected: 逻辑错误 (步子迈大了，或者变量更新顺序错了)
# 5行代码 (行数不同)，且中间算错了
code_rejected = """
def solution(n):
    a = 0
    b = 1
    for i in range(n):
        # 错误：这里直接把 a 覆盖了，导致后续计算错误
        a = b       
        b = a + b   
    return a
"""

# =============================================================================
# 2. 动态追踪工具 (Trace Recorder)
# =============================================================================

class TraceRecorder:
    def __init__(self):
        # 记录格式: List[Dict] -> [{'line': 2, 'vars': {'a': 0, 'b': 1}}, ...]
        self.trace = [] 

    def global_trace(self, frame, event, arg):
        if event == 'call':
            return self.local_trace
        return None

    def local_trace(self, frame, event, arg):
        if event == 'line':
            # 记录当前行号 (相对于函数定义的偏移量)
            # frame.f_code.co_firstlineno 是函数定义的第一行行号
            current_line = frame.f_lineno
            relative_line = current_line - frame.f_code.co_firstlineno
            
            self.trace.append({
                'line': relative_line, # 0-indexed relative line
                'vars': frame.f_locals.copy()
            })
        return self.local_trace

def execute_and_trace(code_str, func_name, *args):
    """
    动态编译并执行代码，返回 Trace 和 结果
    """
    # 1. 编译代码到局部命名空间
    local_scope = {}
    try:
        exec(textwrap.dedent(code_str), {}, local_scope)
    except Exception as e:
        print(f"Compilation Error: {e}")
        return [], None

    func = local_scope[func_name]
    
    # 2. 注入追踪器
    recorder = TraceRecorder()
    sys.settrace(recorder.global_trace)
    
    try:
        res = func(*args)
    except Exception as e:
        res = f"Error: {e}"
    finally:
        sys.settrace(None) # 务必关闭
        
    return recorder.trace, res

# =============================================================================
# 3. 核心策略: 寻找分歧点并生成掩码
# =============================================================================

def find_divergence_and_mask(trace_c, trace_r, rejected_code_str):
    """
    对比两个 Trace，找到 Rejected 中第一个出错的行，并生成掩码。
    """
    divergence_step = -1
    divergence_line_in_rejected = -1
    diff_vars = {}

    # 按步骤 (Time Step) 对比
    # 我们只对比两个 Trace 的公共长度部分
    min_len = min(len(trace_c), len(trace_r))
    
    print(f"{'Step':<5} | {'Chosen State':<30} | {'Rejected State':<30} | {'Status'}")
    print("-" * 90)

    for i in range(min_len):
        state_c = trace_c[i]
        state_r = trace_r[i]
        
        vars_c = state_c['vars']
        vars_r = state_r['vars']
        
        # 检查关键变量是否一致
        is_match = True
        current_diff = []
        
        for key in vars_c:
            # 只对比 Rejected 中也存在的变量 (忽略 Rejected 还没定义或名字不一样的)
            # 严格模式：如果 Rejected 少了 Chosen 有的变量，也可以视为错
            if key in vars_r:
                # 注意：浮点数对比可能需要 epsilon，这里演示用整数
                if vars_c[key] != vars_r[key]:
                    is_match = False
                    current_diff.append(f"{key}: {vars_c[key]}!={vars_r[key]}")
        
        status = "OK" if is_match else f"MISMATCH! {current_diff}"
        print(f"{i:<5} | {str(vars_c):<30} | {str(vars_r):<30} | {status}")
        
        if not is_match:
            divergence_step = i
            divergence_line_in_rejected = state_r['line']
            break
            
    # ==========================================
    # 生成 Mask (Saturated Weighting Strategy)
    # ==========================================
    lines = rejected_code_str.strip().split('\n')
    total_lines = len(lines)
    
    # 默认全 0.1 (保护正确前缀)
    mask_weights = [0.1] * total_lines
    
    if divergence_line_in_rejected != -1:
        print(f"\n>>> First divergence at Step {divergence_step}, Rejected Line {divergence_line_in_rejected}")
        print(f">>> Content: {lines[divergence_line_in_rejected].strip()}")
        
        # 策略：从出错行开始，直到结束，全部设为 1.0 (惩罚)
        for idx in range(divergence_line_in_rejected, total_lines):
            mask_weights[idx] = 1.0
    else:
        print("\n>>> No divergence found in trace (Logic might be correct for this input)")
    
    return mask_weights

# =============================================================================
# 4. 运行测试
# =============================================================================

if __name__ == "__main__":
    # 测试输入
    test_input = 5
    
    print(f"Running Chosen Code (Input={test_input})...")
    trace_chosen, res_chosen = execute_and_trace(code_chosen, "solution", test_input)
    print(f"Result: {res_chosen}\n")
    
    print(f"Running Rejected Code (Input={test_input})...")
    trace_rejected, res_rejected = execute_and_trace(code_rejected, "solution", test_input)
    print(f"Result: {res_rejected}\n")
    
    print("Analyzing Trace Differences...")
    mask = find_divergence_and_mask(trace_chosen, trace_rejected, code_rejected)
    
    print("\n=== Final Mask Application ===")
    lines = code_rejected.strip().split('\n')
    for i, (line, weight) in enumerate(zip(lines, mask)):
        tag = " [PENALIZE -->]" if weight == 1.0 else " [PROTECT]"
        print(f"Line {i:<2} (w={weight:.1f}): {line:<30} {tag}")