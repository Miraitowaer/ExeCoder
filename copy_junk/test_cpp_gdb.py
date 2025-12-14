import os
import subprocess
import json
import sys

# ==========================================
# 1. 准备测试数据 (Case: 只有 main 函数的逻辑)
# ==========================================
# 这是一个典型的 "All in Main" 场景，之前的脚本会失败，这个版本能过。
cpp_chosen = """
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n = 5;
    int a = 0, b = 1;
    // 斐波那契数列逻辑直接写在 main 里
    for (int i = 0; i < n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    // cout << a << endl;
    return 0;
}
"""

# Rejected: 逻辑错误 (temp 计算错)
cpp_rejected = """
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n = 5;             // Line 6
    int a = 0, b = 1;      // Line 7
    for (int i = 0; i < n; ++i) { // Line 8
        int temp = a - b;  // Line 9 (ERROR: Should be +)
        a = b;             // Line 10
        b = temp;          // Line 11
    }
    return 0;
}
"""

# ==========================================
# 2. 工具函数
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GDB_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "gdb_trace.py")

def run_command(cmd):
    subprocess.check_call(cmd, shell=True)

def get_cpp_trace(binary_name, source_code):
    src_file = f"{binary_name}.cpp"
    with open(src_file, "w") as f:
        f.write(source_code)
    
    if os.path.exists(binary_name):
        os.remove(binary_name)
    
    # -g: 必须，否则 GDB 看不到行号
    # -O0: 必须，否则变量会被优化掉
    run_command(f"g++ -g -O0 {src_file} -o {binary_name}")
    
    # 传入源文件名给 GDB 脚本，用于过滤
    # 格式: gdb ... -ex "py SOURCE_FILE='filename.cpp'" ...
    cmd = f"gdb -batch -ex \"py SOURCE_FILE='{src_file}'\" -x {GDB_SCRIPT_PATH} -ex trace_run ./{binary_name}"
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    trace = []
    for line in stdout.splitlines():
        if line.startswith("JSON_TRACE: "):
            try:
                raw_json = line.replace("JSON_TRACE: ", "", 1)
                data = json.loads(raw_json)
                if "error" not in data and "status" not in data:
                    trace.append(data)
            except: pass
    return trace

# ==========================================
# 3. 核心：Trace 对比 (无需再过滤函数名)
# ==========================================

def find_logic_error(trace_gold, trace_bad):
    print(f"\n{'Step':<5} | {'Line (Rej)':<10} | {'Chosen Vars':<30} | {'Rejected Vars':<30} | {'Status'}")
    print("-" * 110)
    
    # 此时 trace 已经是纯净的了 (只包含 source_file 里的行)
    # 无需再做函数名过滤，直接对齐
    
    min_len = min(len(trace_gold), len(trace_bad))
    
    for i in range(min_len):
        step_c = trace_gold[i]
        step_r = trace_bad[i]
        
        vars_c = step_c['vars']
        vars_r = step_r['vars']
        line_r = step_r['line']
        
        diffs = []
        for k in vars_c:
            if k in vars_r:
                val_c = str(vars_c[k]).split()[0] # 清理 "10 (0xa)"
                val_r = str(vars_r[k]).split()[0]
                if val_c != val_r:
                    diffs.append(k)
        
        status = "OK"
        if diffs:
            status = f"MISMATCH: {diffs}"
        
        # 打印前几步和错误步，避免刷屏
        if i < 3 or diffs:
            print(f"{i:<5} | {line_r:<10} | {str(vars_c):<30} | {str(vars_r):<30} | {status}")
        
        if diffs:
            # 回溯一步定位错误源头
            if i > 0:
                blame_step = trace_bad[i-1]
                return blame_step['line'], diffs, i-1
            return line_r, diffs, i

    return -1, [], -1

# ==========================================
# 4. 主流程
# ==========================================

if __name__ == "__main__":
    # 生成 GDB 内部脚本
    # 关键改进：增加了基于 SOURCE_FILE 变量的文件名过滤逻辑
    gdb_script_content = """
import gdb
import json
import os

class TraceCommand(gdb.Command):
    def __init__(self):
        super(TraceCommand, self).__init__("trace_run", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        gdb.execute("set pagination off")
        gdb.execute("set confirm off")
        
        # 获取外部传入的目标源文件名 (用于过滤)
        # 如果没传，默认不过滤 (危险，会包含系统库)
        try:
            target_file = gdb.parse_and_eval("SOURCE_FILE").string()
            # 只取文件名，忽略路径差异
            target_file = os.path.basename(target_file)
        except:
            target_file = None

        try:
            gdb.execute("start", to_string=True)
        except gdb.error:
            return

        while True:
            try:
                frame = gdb.newest_frame()
                sal = frame.find_sal()
                
                # --- 核心过滤逻辑 ---
                # 1. 行号必须有效
                # 2. 如果指定了 target_file，则当前栈帧必须属于该文件
                is_user_code = False
                if sal.line > 0:
                    if target_file:
                        if sal.symtab and sal.symtab.filename:
                            current_file = os.path.basename(sal.symtab.filename)
                            if current_file == target_file:
                                is_user_code = True
                    else:
                        is_user_code = True # 没指定文件，全抓

                if is_user_code:
                    vars_dict = {}
                    try:
                        block = frame.block()
                        while block:
                            if not block.is_global:
                                for symbol in block:
                                    if symbol.is_variable or symbol.is_argument:
                                        name = symbol.name
                                        if name not in vars_dict:
                                            try:
                                                val = symbol.value(frame)
                                                vars_dict[name] = str(val)
                                            except: pass
                            block = block.superblock
                    except: pass

                    trace_data = {"line": sal.line, "func": frame.name(), "vars": vars_dict}
                    print("JSON_TRACE: " + json.dumps(trace_data))

                # 永远 Step Into，GDB 会自动进入/跳出
                # 配合上面的 is_user_code 过滤，我们只在用户代码里 print，但会穿越系统代码
                gdb.execute("step", to_string=True)
                
            except gdb.error:
                break
            except Exception:
                break
        gdb.execute("quit")

TraceCommand()
"""
    with open(GDB_SCRIPT_PATH, "w") as f:
        f.write(gdb_script_content)

    try:
        print("Tracing Chosen...")
        trace_c = get_cpp_trace("cpp_chosen", cpp_chosen)
        
        print("Tracing Rejected...")
        trace_r = get_cpp_trace("cpp_rejected", cpp_rejected)
        
        if not trace_c or not trace_r:
            print("Error: Trace failed.")
        else:
            print("\nAnalyzing...")
            error_line, diff_vars, step = find_logic_error(trace_c, trace_r)
            
            if error_line != -1:
                print(f"\n[SUCCESS] Detected Logic Error!")
                print(f" -> Mismatch at Step {step+1}")
                print(f" -> Blame Line {error_line}")
                print(f" -> Variable {diff_vars} diverged.")
                
                lines = cpp_rejected.split('\n')
                # C++ line 1-indexed -> Python 0-indexed
                if 0 < error_line <= len(lines):
                    print(f" -> Culprit Code: \"{lines[error_line-1].strip()}\"")
            else:
                print("\n[FAIL] No divergence found.")

    except Exception as e:
        print(f"\nExecution Failed: {e}")
    finally:
        for f in ["cpp_chosen", "cpp_chosen.cpp", "cpp_rejected", "cpp_rejected.cpp", "cpp_chosen.dSYM", "cpp_rejected.dSYM", GDB_SCRIPT_PATH]:
            if os.path.exists(f): os.remove(f)