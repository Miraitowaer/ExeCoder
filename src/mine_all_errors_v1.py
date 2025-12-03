import os
import sys
import json
import re
import tempfile
import subprocess
import argparse
import glob
import textwrap
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 1. 基础设施配置
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JAVA_RUNNER_SRC = os.path.join(SCRIPT_DIR, "TraceRunner.java")
GDB_SCRIPT_SRC = os.path.join(SCRIPT_DIR, "gdb_trace.py")

def find_tools_jar():
    try:
        output = subprocess.check_output("java -XshowSettings:properties -version 2>&1", shell=True, text=True)
        java_home = ""
        for line in output.splitlines():
            if "java.home =" in line:
                java_home = line.split("=")[1].strip()
                break
        candidates = []
        if java_home:
            candidates.extend([
                os.path.join(java_home, "..", "lib", "tools.jar"),
                os.path.join(java_home, "lib", "tools.jar"),
            ])
        candidates.extend([
            "/usr/lib/jvm/default-java/lib/tools.jar",
            "/usr/lib/jvm/java-8-openjdk-amd64/lib/tools.jar"
        ])
        for path in candidates:
            if os.path.exists(path): return os.path.abspath(path)
    except: pass
    return None

def setup_tools():
    runner_class = os.path.join(SCRIPT_DIR, "TraceRunner.class")
    if not os.path.exists(runner_class):
        print(f"[Setup] Compiling TraceRunner.java...")
        tools_jar = find_tools_jar()
        cp = f".:{tools_jar}" if tools_jar else "."
        if os.name == 'nt': cp = f".;{tools_jar}"
        try:
            subprocess.run(f"javac -cp \"{cp}\" -g {JAVA_RUNNER_SRC}", shell=True, check=True)
        except subprocess.CalledProcessError:
            print("[Warning] Failed to compile TraceRunner. Java tracing may fail.")

# ==============================================================================
# 2. 核心组件：追踪器 (The Tracers)
# ==============================================================================

class Tracer:
    def __init__(self):
        self.tools_jar = find_tools_jar()

    def run(self, code, lang, temp_dir, tag=""):
        """返回: (trace_list, output_string)"""
        if not code or not code.strip(): return None, ""
        try:
            if lang == "cpp": return self._run_cpp(code, temp_dir, tag)
            if lang == "java": return self._run_java(code, temp_dir, tag)
            if lang == "python": return self._run_python(code, temp_dir, tag)
        except Exception:
            return None, ""
        return None, ""

    def _parse_output_and_trace(self, raw_stdout):
        trace = []
        user_output_lines = []
        
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line: continue
            
            is_trace = False
            if line.startswith("JSON_TRACE:"):
                try: 
                    json_str = line.split("JSON_TRACE:", 1)[1].strip()
                    trace.append(json.loads(json_str))
                    is_trace = True
                except: pass
            elif line.startswith("{") and "vars" in line:
                try: 
                    trace.append(json.loads(line))
                    is_trace = True
                except: pass
            
            if not is_trace:
                user_output_lines.append(line)
                
        return trace, "\n".join(user_output_lines)

    def _run_cpp(self, code, temp_dir, tag):
        filename = f"prog_{tag}.cpp"
        src = os.path.join(temp_dir, filename)
        exe = os.path.join(temp_dir, f"prog_{tag}")
        with open(src, "w") as f: f.write(code)
        
        try:
            subprocess.run(f"g++ -g -O0 -w {src} -o {exe}", shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return None, ""

        gdb_cmd = [
            "gdb", "-batch",
            "-ex", f"py SOURCE_FILE='{filename}'", 
            "-x", GDB_SCRIPT_SRC,
            "-ex", "trace_run " + exe
        ]
        try:
            res = subprocess.run(gdb_cmd, cwd=temp_dir, capture_output=True, text=True, timeout=10)
            return self._parse_output_and_trace(res.stdout)
        except subprocess.TimeoutExpired:
            return None, ""

    def _prepare_java_code(self, code, temp_dir, tag=""):
        code = re.sub(r'^\s*package\s+[\w\.]+;', '', code, flags=re.MULTILINE)
        m = re.search(r'public\s+class\s+(\w+)', code)
        if m: cls_name = m.group(1)
        else:
            m = re.search(r'\bclass\s+(\w+)', code)
            if m: cls_name = m.group(1)
            else: cls_name = "Main"
        src_path = os.path.join(temp_dir, f"{cls_name}.java")
        with open(src_path, "w", encoding='utf-8') as f: f.write(code)
        return cls_name, src_path

    def _run_java(self, code, temp_dir, tag):
        try:
            cls_name, src_path = self._prepare_java_code(code, temp_dir, tag)
            subprocess.run(f"javac -g -nowarn {src_path}", shell=True, cwd=temp_dir, check=True, capture_output=True)
            
            runner_dir = SCRIPT_DIR
            cp_parts = [".", runner_dir]
            if self.tools_jar: cp_parts.insert(0, self.tools_jar)
            sep = ";" if os.name == 'nt' else ":"
            cp = sep.join(cp_parts)
            
            cmd = f"java -cp \"{cp}\" TraceRunner {cls_name}"
            res = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True, timeout=10)
            return self._parse_output_and_trace(res.stdout)
        except:
            return None, ""

    def _run_python(self, code, temp_dir, tag):
        script_name = f"script_{tag}.py"
        script_path = os.path.join(temp_dir, script_name)
        with open(script_path, "w") as f: f.write(code)
        
        driver_name = f"driver_{tag}.py"
        driver_path = os.path.join(temp_dir, driver_name)
        
        driver_code = f"""
import sys, json, os
trace_data = []
def tracer(frame, event, arg):
    if event == 'line' and frame.f_code.co_filename.endswith("{script_name}"):
        try:
            vars_snap = {{k:str(v) for k,v in frame.f_locals.items() if not k.startswith('__')}}
            trace_data.append({{'line': frame.f_lineno, 'vars': vars_snap}})
        except: pass
    return tracer

with open("{script_name}", "r") as f:
    code_obj = compile(f.read(), "{script_name}", "exec")

sys.settrace(tracer)
try:
    exec(code_obj, {{'__name__': '__main__'}})
except Exception: pass 
finally: sys.settrace(None)

for item in trace_data: print(json.dumps(item))
"""
        with open(driver_path, "w") as f: f.write(driver_code)
        try:
            res = subprocess.run([sys.executable, driver_name], cwd=temp_dir, capture_output=True, text=True, timeout=10)
            return self._parse_output_and_trace(res.stdout)
        except:
            return None, ""

# ==============================================================================
# 3. 工作流：ErrorMiner
# ==============================================================================

class ErrorMiner:
    def __init__(self):
        self.tracer = Tracer()
        self.ignore_keys = {'args', 'clazz', 'method', 'this', 'self', 'module', '__name__', '__file__', '__builtins__', '__doc__', '__loader__', '__package__', '__spec__'}

    def detect_lang(self, item):
        text = (item.get('instruction', '') + " " + item.get('input', '')).lower()
        match = re.search(r"\bto\s+(c\+\+|cpp|java|python)\b", text)
        if match:
            target = match.group(1)
            if target in ["c++", "cpp"]: return "cpp"
            if target == "java": return "java"
            if target == "python": return "python"
        code = item.get('output', '') 
        if not code: return None
        if "public class" in code or ("class " in code and "public static void main" in code): return "java"
        if "#include" in code or "using namespace" in code: return "cpp"
        if "def " in code and ":" in code and "import " in code: return "python"
        return None

    def _calculate_line_offset(self, raw_text):
        if "```" in raw_text:
            pattern = r"```(?:\w+)?\n(.*?)```"
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                code_start_index = match.start(1)
                prefix = raw_text[:code_start_index]
                return prefix.count('\n')
        return 0

    def process(self, item):
        lang = self.detect_lang(item)
        if not lang: return None

        prompt = item.get('instruction', '') + "\n" + item.get('input', '')
        chosen = item.get('output', '')
        rejected = item.get('generated_output', '')
        
        chosen_pure = self._extract_code(chosen)
        rejected_pure = self._extract_code(rejected)
        
        if not chosen_pure or not rejected_pure: return None
        offset = self._calculate_line_offset(rejected)

        result = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "lang": lang,
            "error_type": None,
            "error_lines": [],
            "error_msg": ""
        }

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 1. 运行 Rejected
                trace_r, out_r = self.tracer.run(rejected_pure, lang, tmp_dir, "rej")
                
                if not trace_r:
                    is_valid, err_msg = self._check_compile(rejected_pure, lang, tmp_dir)
                    if not is_valid:
                        result['error_type'] = "compilation_error"
                        result['error_msg'] = err_msg
                        rel_lines = self._parse_compiler_error_lines(err_msg, lang)
                        result['error_lines'] = [l + offset for l in rel_lines]
                    else:
                        result['error_type'] = "execution_error"
                        result['error_msg'] = "Runtime Crash or No Output"
                        result['error_lines'] = [0 + offset] 
                    return result

                # 2. 运行 Chosen
                trace_c, out_c = self.tracer.run(chosen_pure, lang, tmp_dir, "gold")
                if not trace_c: return None 

                # 3. Output Guard (结果一致则认为正确)
                if out_c.strip() == out_r.strip():
                    return None
                
                # 4. Trace Contrast (结果不同，寻找分歧点)
                divergence_idx = self._diff_trace_linear(trace_c, trace_r)
                
                if divergence_idx != -1:
                    
                    # ==========================================================
                    # [UPDATED] 饱和式惩罚策略 (Saturated Penalization)
                    # ==========================================================
                    # 1. 回溯一步 (Backtrack): 惩罚分歧的前一步 (Step K-1)
                    # 2. 当前步 (Current): 惩罚分歧点 (Step K)
                    # 3. 后续全罚 (Forward Propagation): 惩罚分歧点之后的所有步骤 (Step K+1 ... End)
                    #    理由: 一旦逻辑出错，后续执行路径都是基于错误状态的，都应被视为 Invalid。
                    
                    error_lines_set = set()
                    
                    # (1) Backtrack Step
                    if divergence_idx > 0:
                        prev_step = trace_r[divergence_idx - 1]
                        error_lines_set.add(prev_step.get('line', 1))
                        
                    # (2) & (3) Current Step + All Future Steps
                    # 遍历 Rejected Trace 从 divergence_idx 开始直到结束的所有步骤
                    for i in range(divergence_idx, len(trace_r)):
                        step = trace_r[i]
                        error_lines_set.add(step.get('line', 1))

                    # 转为绝对行号
                    result['error_lines'] = sorted([l - 1 + offset for l in error_lines_set])
                    result['error_type'] = "logic_error"
                    
                    diff_log = self._format_trace_comparison(trace_c, trace_r, divergence_idx)
                    result['error_msg'] = f"Output Mismatch:\n[Chosen]: {out_c.strip()}\n[Rejected]: {out_r.strip()}\n\n{diff_log}"
                    return result
                
                # Trace 没差异但 Output 有差异 (Execution Error)
                result['error_type'] = "execution_error"
                result['error_lines'] = [0 + offset]
                result['error_msg'] = f"Output mismatch but identical trace prefix. \nChosen: {out_c} \nRejected: {out_r}"
                return result

        except Exception:
            return None

    def _extract_code(self, text):
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else text

    def _check_compile(self, code, lang, tmp_dir):
        try:
            if lang == "python":
                f = os.path.join(tmp_dir, "check.py"); open(f,"w").write(code)
                res = subprocess.run([sys.executable, "-m", "py_compile", f], capture_output=True, text=True)
                return res.returncode == 0, res.stderr
            elif lang == "java":
                code = re.sub(r'^\s*package\s+[\w\.]+;', '', code, flags=re.MULTILINE)
                m = re.search(r'public\s+class\s+(\w+)', code)
                if m: cls_name = m.group(1)
                else:
                    m = re.search(r'\bclass\s+(\w+)', code)
                    cls_name = m.group(1) if m else "Main"
                f = os.path.join(tmp_dir, f"{cls_name}.java")
                with open(f, "w", encoding='utf-8') as fp: fp.write(code)
                res = subprocess.run(["javac", "-nowarn", f], capture_output=True, text=True)
                return res.returncode == 0, res.stderr
            elif lang == "cpp":
                f = os.path.join(tmp_dir, "check.cpp"); open(f,"w").write(code)
                res = subprocess.run(["g++", "-fsyntax-only", "-w", f], capture_output=True, text=True)
                return res.returncode == 0, res.stderr
        except: return False, "Compiler Crash"
        return True, ""

    def _parse_compiler_error_lines(self, error_msg, lang):
        lines = set()
        if not error_msg: return []
        global_indicators = [
            r"cannot find symbol", r"package .* does not exist", r"import .* not found",
            r"was not declared", r"no such file or directory", r"ModuleNotFoundError",
            r"ImportError", r"symbol: class", r"is defined in header", r"did you forget to"
        ]
        for pattern in global_indicators:
            if re.search(pattern, error_msg, re.IGNORECASE):
                return []
        if lang == 'cpp': 
            matches = re.findall(r':(\d+):\d+: error:', error_msg)
            for m in matches: lines.add(int(m) - 1)
        elif lang == 'java':
            matches = re.findall(r':(\d+): error:', error_msg)
            for m in matches: lines.add(int(m) - 1)
        elif lang == 'python':
            matches = re.findall(r'line\s+(\d+)', error_msg)
            for m in matches: lines.add(int(m) - 1)
        return sorted(list(lines))

    def _diff_trace_linear(self, tc, tr):
        min_len = min(len(tc), len(tr))
        for i in range(min_len):
            vc = tc[i].get('vars', {})
            vr = tr[i].get('vars', {})
            for k in vr:
                if k in self.ignore_keys: continue
                if k in vc:
                    val_c = str(vc[k]).strip().split()[0]
                    val_r = str(vr[k]).strip().split()[0]
                    if val_c != val_r:
                        return i
        return -1

    def _format_trace_comparison(self, tc, tr, error_idx):
        lines = []
        header = f"{'Step':<5} | {'Chosen Vars':<30} | {'Rejected Vars':<30} | {'Status'}"
        lines.append(header)
        lines.append("-" * 100)

        start = max(0, error_idx - 3)
        end = min(min(len(tc), len(tr)), error_idx + 2)
        
        for i in range(start, end):
            vc = tc[i].get('vars', {})
            vr = tr[i].get('vars', {})
            
            vc_str = str({k: v for k, v in vc.items() if k not in self.ignore_keys})
            vr_str = str({k: v for k, v in vr.items() if k not in self.ignore_keys})
            
            status = "OK"
            if i == error_idx:
                status = "MISMATCH <--- [ERROR]"
            elif i > error_idx:
                status = "DIVERGED"
            
            lines.append(f"{i:<5} | {vc_str} | {vr_str} | {status}")
        
        return "\n".join(lines)

# ==============================================================================
# 4. Main Entry
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./data/dpo_mining")
    parser.add_argument('--output_file', type=str, default="./data/dpo_train_all.json")
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    setup_tools()
    
    files = glob.glob(os.path.join(args.input_dir, "dpo_sample_*_debug.json"))
    all_samples = []
    for f in files:
        try: 
            content = json.load(open(f))
            if isinstance(content, list): all_samples.extend(content)
            else: all_samples.append(content)
        except: pass
        
    print(f"Loaded {len(all_samples)} samples.")
    
    miner = ErrorMiner()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(miner.process, item) for item in all_samples]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = fut.result()
                if res: results.append(res)
            except: pass
            
    print(f"Mining complete. Found {len(results)} errors.")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()