import os
import subprocess
import json
import sys

# ==========================================
# 1. 准备测试数据 (包含函数调用，测试 STEP_INTO)
# ==========================================
java_chosen = """
public class SolutionChosen {
    public static void main(String[] args) {
        int res = calculate(5, 3);
    }

    public static int calculate(int a, int b) {
        int x = a * 2;
        int y = b * 2;
        int sum = x + y; // Correct Logic
        return sum;
    }
}
"""

java_rejected = """
public class SolutionRejected {
    public static void main(String[] args) {
        int res = calculate(5, 3);
    }

    public static int calculate(int a, int b) {
        int x = a * 2;   // Line 9
        int y = b * 2;   // Line 10
        int sum = x - y; // Line 11 (ERROR: Should be +)
        return sum;      // Line 12
    }
}
"""

# ==========================================
# 2. 工具函数 (保持不变，为了完整性再次列出)
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_tools_jar():
    try:
        output = subprocess.check_output("java -XshowSettings:properties -version 2>&1", shell=True, text=True)
        java_home = ""
        for line in output.splitlines():
            if "java.home =" in line:
                java_home = line.split("=")[1].strip()
                break
        if not java_home: return None
        candidates = [
            os.path.join(java_home, "..", "lib", "tools.jar"),
            os.path.join(java_home, "lib", "tools.jar"),
            "/usr/lib/jvm/default-java/lib/tools.jar",
            "/usr/lib/jvm/java-8-openjdk-amd64/lib/tools.jar"
        ]
        for path in candidates:
            if os.path.exists(path): return os.path.abspath(path)
    except: pass
    return None

def run_command(cmd):
    # print(f"[CMD] {cmd}") 
    subprocess.check_call(cmd, shell=True)

def get_java_trace(class_name, source_code, tools_jar):
    file_name = f"{class_name}.java"
    with open(file_name, "w") as f: f.write(source_code)
    
    run_command(f"javac -g {file_name}")
    
    runner_src = os.path.join(SCRIPT_DIR, "TraceRunner.java")
    runner_class = os.path.join(SCRIPT_DIR, "TraceRunner.class")
    
    if not os.path.exists(runner_class):
        print("Compiling TraceRunner...")
        cp_compile = f".:{tools_jar}" if tools_jar else "."
        run_command(f"javac -cp \"{cp_compile}\" -g {runner_src}")
    
    print(f"Tracing {class_name} (Step-Into mode)...")
    separator = ";" if os.name == 'nt' else ":"
    classpath_parts = [SCRIPT_DIR, "."]
    if tools_jar: classpath_parts.insert(0, tools_jar)
    classpath = separator.join(classpath_parts)
    
    cmd = f"java -cp \"{classpath}\" TraceRunner {class_name}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    trace = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try: trace.append(json.loads(line))
            except: pass
    return trace

# ==========================================
# 3. 核心：精准定位 (Backtracking Strategy)
# ==========================================

def find_precise_error(trace_gold, trace_bad):
    print(f"\n{'Step':<5} | {'Line (Rej)':<10} | {'Chosen Vars':<30} | {'Rejected Vars':<30} | {'Status'}")
    print("-" * 110)

    min_len = min(len(trace_gold), len(trace_bad))
    
    for i in range(min_len):
        step_c = trace_gold[i]
        step_r = trace_bad[i]
        
        vars_c = step_c.get('vars', {})
        vars_r = step_r.get('vars', {})
        line_r = step_r['line']
        
        diffs = []
        for k in vars_c:
            if k in vars_r and vars_c[k] != vars_r[k]:
                diffs.append(k)
        
        status = "OK"
        if diffs:
            status = f"MISMATCH: {diffs}"
        
        print(f"{i:<5} | {line_r:<10} | {str(vars_c):<30} | {str(vars_r):<30} | {status}")
        
        if diffs:
            # --- 关键修正：回溯一步 ---
            # 如果当前步(i)变量变了，说明是上一步(i-1)执行的代码导致的。
            # 也就是 Trace 中上一个记录的行号才是"作案现场"。
            if i > 0:
                blame_step = trace_bad[i-1]
                blame_line = blame_step['line']
                return blame_line, diffs, i-1
            else:
                # 极罕见情况：第一行变量就错了（可能是参数传错）
                return line_r, diffs, i

    return -1, [], -1

# ==========================================
# 4. 主流程
# ==========================================

if __name__ == "__main__":
    for f in ["SolutionChosen.class", "SolutionRejected.class", "SolutionChosen.java", "SolutionRejected.java"]:
        if os.path.exists(f): os.remove(f)

    try:
        tools_jar = find_tools_jar()
        trace_c = get_java_trace("SolutionChosen", java_chosen, tools_jar)
        trace_r = get_java_trace("SolutionRejected", java_rejected, tools_jar)
        
        if not trace_c or not trace_r:
            print("Error: Trace failed.")
        else:
            error_line, diff_vars, step_idx = find_precise_error(trace_c, trace_r)
            
            if error_line != -1:
                print(f"\n[SUCCESS] Precise Error Localization!")
                print(f" -> Mismatch detected at Step {step_idx+1} (Line {trace_r[step_idx+1]['line']})")
                print(f" -> Blame assigned to Step {step_idx} (Line {error_line})")
                
                lines = java_rejected.split('\n')
                if 0 < error_line <= len(lines):
                    code_content = lines[error_line-1].strip()
                    print(f" -> Culprit Code: \"{code_content}\"")
                    
                    # 验证是否精准定位到了 sum = x - y
                    if "sum = x - y" in code_content:
                        print(" -> VERDICT: PERFECT MATCH! 🎯")
                    else:
                        print(" -> VERDICT: Near match (Context difference)")
            else:
                print("\n[FAIL] No divergence found.")
            
    except Exception as e:
        print(f"\nExecution Failed: {e}")