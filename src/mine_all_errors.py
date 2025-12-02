import os
import sys
import json
import re
import tempfile
import subprocess
import argparse
import shutil
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 1. 基础设施配置 & 嵌入式辅助脚本
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JAVA_RUNNER_SRC = os.path.join(SCRIPT_DIR, "TraceRunner.java")
GDB_SCRIPT_SRC = os.path.join(SCRIPT_DIR, "gdb_trace.py")

def setup_tools():
    """确保 TraceRunner.java 和 gdb_trace.py 存在于 src 目录"""
    
    # 1. 嵌入 GDB 脚本
    if not os.path.exists(GDB_SCRIPT_SRC):
        with open(GDB_SCRIPT_SRC, "w") as f:
            f.write("""
import gdb, json, os
class TraceCommand(gdb.Command):
    def __init__(self): super(TraceCommand, self).__init__("trace_run", gdb.COMMAND_USER)
    def invoke(self, arg, from_tty):
        gdb.execute("set pagination off"); gdb.execute("set confirm off")
        try: target = gdb.parse_and_eval("SOURCE_FILE").string(); target=os.path.basename(target)
        except: target = None
        try: gdb.execute("start", to_string=True)
        except: return
        while True:
            try:
                f = gdb.newest_frame(); sal = f.find_sal()
                if sal.line > 0:
                    valid = True
                    if target and sal.symtab and sal.symtab.filename:
                        if os.path.basename(sal.symtab.filename) != target: valid = False
                    if valid:
                        vs = {}
                        try:
                            b = f.block()
                            while b:
                                if not b.is_global:
                                    for s in b:
                                        if (s.is_variable or s.is_argument) and s.name not in vs:
                                            try: 
                                                val = str(s.value(f))
                                                if "=" in val: val = val.split("=")[-1].strip() # Clean C++ output
                                                vs[s.name] = val.split()[0] 
                                            except: pass
                                b = b.superblock
                        except: pass
                        print("JSON_TRACE: " + json.dumps({"line": sal.line, "vars": vs}))
                gdb.execute("step", to_string=True)
            except: break
        gdb.execute("quit")
TraceCommand()
""")

    # 2. 嵌入 Java JDI Runner
    if not os.path.exists(JAVA_RUNNER_SRC):
        with open(JAVA_RUNNER_SRC, "w") as f:
            f.write("""
import com.sun.jdi.*;
import com.sun.jdi.connect.*;
import com.sun.jdi.event.*;
import com.sun.jdi.request.*;
import java.util.List;
import java.util.Map;

public class TraceRunner {
    public static void main(String[] args) throws Exception {
        String targetClass = args[0];
        LaunchingConnector connector = Bootstrap.virtualMachineManager().defaultConnector();
        Map<String, Connector.Argument> arguments = connector.defaultArguments();
        arguments.get("main").setValue(targetClass);
        arguments.get("options").setValue("-cp .");
        VirtualMachine vm = connector.launch(arguments);
        vm.eventRequestManager().createClassPrepareRequest().enable();
        EventQueue q = vm.eventQueue();
        boolean connected = true;
        while (connected) {
            EventSet es = q.remove();
            for (Event e : es) {
                if (e instanceof ClassPrepareEvent) {
                    ReferenceType rt = ((ClassPrepareEvent) e).referenceType();
                    if (rt.name().equals(targetClass)) {
                        MethodEntryRequest mer = vm.eventRequestManager().createMethodEntryRequest();
                        mer.addClassFilter(targetClass);
                        mer.enable();
                    }
                }
                if (e instanceof MethodEntryEvent) {
                    MethodEntryEvent mee = (MethodEntryEvent) e;
                    if (mee.method().name().equals("main")) {
                        StepRequest sr = vm.eventRequestManager().createStepRequest(mee.thread(), StepRequest.STEP_LINE, StepRequest.STEP_INTO);
                        sr.addClassFilter(targetClass);
                        sr.enable();
                    }
                }
                if (e instanceof StepEvent) {
                    try {
                        StackFrame f = ((StepEvent) e).thread().frame(0);
                        System.out.print("{\\"line\\": " + f.location().lineNumber() + ", \\"vars\\": {");
                        boolean first = true;
                        for (LocalVariable v : f.visibleVariables()) {
                            Value val = f.getValue(v);
                            if (!first) System.out.print(", ");
                            String s = (val != null) ? val.toString().replace("\\"", "\\\\\\"") : "null";
                            System.out.print("\\"" + v.name() + "\\": \\"" + s + "\\"");
                            first = false;
                        }
                        System.out.println("}}");
                    } catch (Exception x) {}
                }
                if (e instanceof VMDisconnectEvent) connected = false;
            }
            es.resume();
        }
    }
}
""")

    # 3. 预编译 Java Runner
    if not os.path.exists(JAVA_RUNNER_SRC.replace(".java", ".class")):
        try:
            tools_jar = find_tools_jar()
            cp = f".:{tools_jar}" if tools_jar else "."
            if os.name == 'nt': cp = f".;{tools_jar}"
            subprocess.run(f"javac -cp \"{cp}\" -g {JAVA_RUNNER_SRC}", shell=True, check=True)
        except Exception as e:
            print(f"[Warning] Failed to compile TraceRunner: {e}")

def find_tools_jar():
    try:
        out = subprocess.check_output("java -XshowSettings:properties -version 2>&1", shell=True, text=True)
        home = [l.split("=")[1].strip() for l in out.splitlines() if "java.home =" in l][0]
        candidates = [os.path.join(home, "..", "lib", "tools.jar"), os.path.join(home, "lib", "tools.jar"),
                      "/usr/lib/jvm/default-java/lib/tools.jar"]
        for p in candidates: 
            if os.path.exists(p): return p
    except: pass
    return None

# ==============================================================================
# 2. 核心组件：代码移植 (The Transplanter)
# ==============================================================================

class Transplanter:
    @staticmethod
    def process(chosen, rejected, lang):
        try:
            if lang == "cpp": return Transplanter._cpp(chosen, rejected)
            if lang == "java": return Transplanter._java(chosen, rejected)
            if lang == "python": return Transplanter._python(chosen, rejected)
        except Exception as e:
            print(f"  [Transplant Error] {e}")
        return None

    @staticmethod
    def _cpp(chosen, rejected):
        # 1. 提取 Chosen Main
        main_pattern = r'\b(int|void)\s+main\s*\([^)]*\)\s*\{'
        main_match = re.search(main_pattern, chosen, re.DOTALL)
        if not main_match: return None
        chosen_main = chosen[main_match.start():]
        
        # 2. 提取 Rejected Body (切掉它自己的 main)
        rej_main_match = re.search(main_pattern, rejected, re.DOTALL)
        if rej_main_match:
            rejected_body = rejected[:rej_main_match.start()]
        else:
            rejected_body = rejected 

        # 3. 提取并合并 Headers
        headers_c = re.findall(r'^\s*#include.*', chosen, re.MULTILINE)
        headers_r = re.findall(r'^\s*#include.*', rejected, re.MULTILINE)
        usings = re.findall(r'^\s*using\s+namespace.*', chosen, re.MULTILINE)
        
        all_headers = sorted(list(set(headers_c + headers_r + usings)))
        header_block = "\n".join(all_headers) + "\n"
        
        return header_block + "\n" + rejected_body + "\n" + chosen_main

    @staticmethod
    def _java(chosen, rejected):
        # 1. [新增] 移除 package 声明 (防止在 tmp 目录编译失败)
        rejected = re.sub(r'^\s*package\s+[\w\.]+;', '', rejected, flags=re.MULTILINE)
        chosen = re.sub(r'^\s*package\s+[\w\.]+;', '', chosen, flags=re.MULTILINE)

        # 2. 提取 Chosen Main Body
        main_pattern = r'public\s+static\s+void\s+main\s*\([^)]*\)\s*\{'
        start_match = re.search(main_pattern, chosen)
        if not start_match: return None
        
        chosen_main_body = Transplanter._extract_brace_content(chosen, start_match.end() - 1)
        if not chosen_main_body: return None
        chosen_main_body = chosen_main_body[1:-1] # 去掉外层 {}

        # 3. 处理 Rejected
        rejected_clean = re.sub(r'^\s*import\s+[\w\.]+;', '', rejected, flags=re.MULTILINE)
        
        # 提取 Imports
        imports_c = re.findall(r'^\s*import\s+[\w\.]+;', chosen, re.MULTILINE)
        imports_r = re.findall(r'^\s*import\s+[\w\.]+;', rejected, re.MULTILINE)
        import_block = "\n".join(sorted(list(set(imports_c + imports_r)))) + "\n"

        # 找到类结束符
        last_brace_idx = rejected_clean.rfind('}')
        if last_brace_idx == -1: return None
        
        # 废弃旧 Main
        rej_main_match = re.search(main_pattern, rejected_clean)
        if rej_main_match:
            prefix = rejected_clean[:rej_main_match.start()]
            suffix = rejected_clean[rej_main_match.end():]
            rejected_clean = prefix + " private static void _unused_main(String[] args) { " + suffix
            last_brace_idx = rejected_clean.rfind('}')

        new_main = f"\n    public static void main(String[] args) {{ {chosen_main_body} }}\n"
        class_body = rejected_clean[:last_brace_idx] + new_main + "\n}"
        
        return import_block + "\n" + class_body

    @staticmethod
    def _python(chosen, rejected):
        driver_match = re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', chosen)
        if driver_match:
            driver = chosen[driver_match.start():]
        else:
            return None 
            
        rej_driver_match = re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', rejected)
        if rej_driver_match:
            body = rejected[:rej_driver_match.start()]
        else:
            body = rejected
            
        return body + "\n\n" + driver

    @staticmethod
    def _extract_brace_content(text, start_brace_index):
        if text[start_brace_index] != '{': return None
        cnt = 0
        for i in range(start_brace_index, len(text)):
            if text[i] == '{': cnt += 1
            elif text[i] == '}':
                cnt -= 1
                if cnt == 0: return text[start_brace_index:i+1]
        return None

# ==============================================================================
# 3. 核心组件：追踪器 (The Tracers)
# ==============================================================================

class Tracer:
    def __init__(self):
        self.tools_jar = find_tools_jar()
        runner_class = JAVA_RUNNER_SRC.replace(".java", ".class")
        if not os.path.exists(runner_class):
            print(f"[Setup] Compiling Java Runner...")
            cp = f".:{self.tools_jar}" if self.tools_jar else "."
            if os.name == 'nt': cp = f".;{self.tools_jar}"
            subprocess.run(f"javac -cp \"{cp}\" -g {JAVA_RUNNER_SRC}", shell=True)

    def run(self, code, lang, temp_dir, tag=""):
        if lang == "cpp": return self._run_cpp(code, temp_dir, tag)
        if lang == "java": return self._run_java(code, temp_dir, tag)
        if lang == "python": return self._run_python(code, temp_dir, tag)
        return None

    # [FIX] 这是一个通用的解析方法，之前漏写了
    def _parse_json_trace(self, output):
        trace = []
        for line in output.splitlines():
            # GDB 格式
            if "JSON_TRACE:" in line:
                try: trace.append(json.loads(line.replace("JSON_TRACE:", "").strip()))
                except: pass
            # Java 格式 (直接是 JSON 行)
            elif line.strip().startswith("{") and "vars" in line:
                try: trace.append(json.loads(line.strip()))
                except: pass
        return trace

    def _run_cpp(self, code, temp_dir, tag):
        src = os.path.join(temp_dir, f"prog_{tag}.cpp")
        exe = os.path.join(temp_dir, f"prog_{tag}")
        with open(src, "w") as f: f.write(code)
        try:
            # 编译
            subprocess.run(f"g++ -g -O0 {src} -o {exe}", shell=True, check=True, capture_output=True)
            # 运行 GDB
            cmd = f"gdb -batch -ex \"py SOURCE_FILE='prog_{tag}.cpp'\" -x {GDB_SCRIPT_SRC} -ex trace_run {exe}"
            res = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True, timeout=5)
            # [FIX] 现在这里可以正确调用了
            return self._parse_json_trace(res.stdout)
        except Exception as e:
            # print(f"[Cpp Error] {e}")
            return None

    def _run_java(self, code, temp_dir, tag):
        m = re.search(r'public\s+class\s+(\w+)', code)
        cls_name = m.group(1) if m else "Main"
        src = os.path.join(temp_dir, f"{cls_name}.java")
        with open(src, "w") as f: f.write(code)
        
        try:
            subprocess.run(f"javac -g {src}", shell=True, cwd=temp_dir, check=True, capture_output=True)
            runner_dir = os.path.dirname(JAVA_RUNNER_SRC)
            cp_parts = [".", runner_dir]
            if self.tools_jar: cp_parts.insert(0, self.tools_jar)
            sep = ";" if os.name == 'nt' else ":"
            cp = sep.join(cp_parts)
            
            cmd = f"java -cp \"{cp}\" TraceRunner {cls_name}"
            res = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True, timeout=10) # 给 Java 多点启动时间
            
            return self._parse_json_trace(res.stdout)
        except subprocess.CalledProcessError as e:
            # 编译失败
            return None
        except Exception as e:
            return None

    def _run_python(self, code, temp_dir, tag):
        # (Python 逻辑保持不变，之前是对的)
        script_name = f"script_{tag}.py"
        driver_name = f"driver_{tag}.py"
        with open(os.path.join(temp_dir, script_name), "w") as f: f.write(code)
        
        driver_code = f"""
import sys, json, os
trace = []
def tracer(frame, event, arg):
    if event == 'line' and frame.f_code.co_filename.endswith("{script_name}"):
        try:
            vars_snap = {{k:str(v) for k,v in frame.f_locals.items() if not k.startswith('__')}}
            trace.append({{'line': frame.f_lineno, 'vars': vars_snap}})
        except: pass
    return tracer
sys.settrace(tracer)
try:
    target_code = open("{script_name}", "r").read()
    global_ns = {{"__name__": "__main__", "__file__": "{script_name}"}}
    exec(target_code, global_ns)
except Exception: pass
finally: sys.settrace(None)
print("JSON_TRACE_START"); print(json.dumps(trace)); print("JSON_TRACE_END")
"""
        with open(os.path.join(temp_dir, driver_name), "w") as f: f.write(driver_code)
        try:
            res = subprocess.run([sys.executable, driver_name], cwd=temp_dir, capture_output=True, text=True, timeout=5)
            if "JSON_TRACE_START" in res.stdout:
                json_part = res.stdout.split("JSON_TRACE_START")[1].split("JSON_TRACE_END")[0]
                return json.loads(json_part)
        except: pass
        return None
    
# ==============================================================================
# 4. 工作流：ErrorMiner
# ==============================================================================

class ErrorMiner:
    def __init__(self):
        self.tracer = Tracer()

    def process(self, item):
        # [DEBUG] 打印正在处理的样本开头
        snippet = item.get('input', '')[:30].replace('\n', ' ')
        print(f"\n[DEBUG] Processing: {snippet}...")

        prompt = item.get('instruction', '') + "\n" + item.get('input', '')
        chosen = item.get('output', '')
        rejected = item.get('generated_output', '')
        
        # 1. 语言检测
        lang = self.detect_lang(item)
        if not lang: 
            print(f"  -> [SKIP] Language detect failed.")
            return None
        print(f"  -> Language: {lang}")

        result = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "error_type": None,
            "error_lines": [],
            "error_msg": ""
        }

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 1. 编译检查
                rejected_pure = self._extract_code(rejected)
                chosen_pure = self._extract_code(chosen)
                
                is_valid, error_msg, raw_line = self._check_compile(rejected_pure, lang, tmp_dir)
                
                if not is_valid:
                    result['error_type'] = "compilation_error"
                    result['error_msg'] = error_msg
                    # 提取行号逻辑
                    relative = self._parse_compiler_error_lines(error_msg, lang)
                    offset = self._calculate_line_offset(rejected)
                    abs_lines = [l + offset for l in relative]
                    # 兜底
                    if not abs_lines and raw_line != -1: abs_lines = [raw_line + offset]
                    result['error_lines'] = abs_lines
                    return result

                # 2. 移花接木
                target_code = Transplanter.process(chosen_pure, rejected_pure, lang)
                if not target_code: return None 

                # 3. 执行 Rejected
                trace_r = self.tracer.run(target_code, lang, tmp_dir, "rej")
                
                # 4. 执行 Chosen (Gold)
                trace_c = self.tracer.run(chosen_pure, lang, tmp_dir, "gold") # 注意：Chosen 不需要 transplant，自带 main
                
                if not trace_c: return None # Gold 跑不通

                # 5. 核心对比逻辑
                # 先看 Logic Error
                diff_line = self._diff_trace(trace_c, trace_r)
                if diff_line != -1:
                    offset = self._calculate_line_offset(rejected)
                    final_line = diff_line - 1 + offset # Trace 1-based -> 0-based
                    
                    result['error_type'] = "logic_error"
                    result['error_lines'] = [final_line]
                    result['error_msg'] = f"Trace divergence at line {final_line}"
                    return result
                
                # 再看 Execution Error (如果 Trace 没差异，但 R 崩了)
                if not trace_r or (len(trace_r)>0 and "error" in trace_r[-1]):
                    result['error_type'] = "execution_error"
                    # 尝试取 Trace 最后一行作为 Crash 地点
                    last_line = trace_r[-1].get('line', 1) - 1 if trace_r else 0
                    result['error_lines'] = [last_line + self._calculate_line_offset(rejected)]
                    result['error_msg'] = "Crash"
                    return result

        except Exception as e:
            pass
            
        return None

    # --- 辅助函数 (保持你的原样，或者把上面的 update 进去) ---

    def detect_lang(self, item):
        full_text = (item.get('instruction', '') + " " + item.get('prompt', '')).lower()
        if re.search(r"to\s+python", full_text): return "python"
        if re.search(r"to\s+java", full_text): return "java"
        if re.search(r"to\s+(cpp|c\+\+)", full_text): return "cpp"
        
        code_snippet = item.get('rejected', '')[:200]
        if "def " in code_snippet: return "python"
        if "#include" in code_snippet: return "cpp"
        if "public class" in code_snippet: return "java"
        return None

    def _extract_code(self, text):
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match: return match.group(1)
        return text

    def _calculate_line_offset(self, raw_text):
        if "```" in raw_text:
            pattern = r"```(?:\w+)?\n(.*?)```"
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                return raw_text[:match.start(1)].count('\n')
        return 0

    def _check_compile(self, code, lang, tmp_dir):
        try:
            if lang == "python":
                f = os.path.join(tmp_dir, "c.py"); open(f,"w").write(code)
                res = subprocess.run([sys.executable, "-m", "py_compile", f], capture_output=True, text=True)
                # Return: success, msg, line_idx_hint
                return (res.returncode == 0), res.stderr, -1
                
            elif lang == "cpp":
                f = os.path.join(tmp_dir, "c.cpp"); open(f,"w").write(code)
                res = subprocess.run(["g++", "-fsyntax-only", "-w", f], capture_output=True, text=True)
                return (res.returncode == 0), res.stderr, -1

            elif lang == "java":
                cls = "Main"
                m = re.search(r'public\s+class\s+(\w+)', code)
                if m: cls = m.group(1)
                f = os.path.join(tmp_dir, f"{cls}.java"); open(f,"w").write(code)
                res = subprocess.run(["javac", f], capture_output=True, text=True)
                return (res.returncode == 0), res.stderr, -1
                
            return True, "", -1
        except Exception as e:
            return False, str(e), -1

    def _parse_compiler_error_lines(self, error_msg, lang):
        lines = set()
        if not error_msg: return []
        global_patterns = [
            r"was not declared", r"did you forget to", r"cannot find symbol",
            r"package .* does not exist", r"ImportError", r"ModuleNotFoundError",
            r"is defined in header", r"symbol: class"
        ]
        for p in global_patterns:
            if re.search(p, error_msg, re.IGNORECASE): return []

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

    def _diff_trace(self, tc, tr):
        if not tc or not tr: return -1
        min_len = min(len(tc), len(tr))
        ignore = {'args', 'clazz', 'method', 'this', 'trace_log', 'trace', 'module'}
        
        for i in range(min_len):
            vc = tc[i].get('vars', {})
            vr = tr[i].get('vars', {})
            for k in vc:
                if k in ignore: continue
                if k in vr:
                    val_c = str(vc[k]).strip().split()[0]
                    val_r = str(vr[k]).strip().split()[0]
                    if val_c != val_r:
                        if i > 0: return tr[i-1].get('line', -1)
                        return tr[i].get('line', -1)
        return -1

# ==============================================================================
# 5. Main Entry
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./data/dpo_mining")
    parser.add_argument('--output_file', type=str, default="./data/dpo_train_all.json")
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    
    setup_tools()
    
    # 收集所有样本文件
    files = glob.glob(os.path.join(args.input_dir, "dpo_sample_*_debug.json"))
    all_samples = []
    for f in files:
        try: all_samples.extend(json.load(open(f)))
        except: pass
        
    print(f"Loaded {len(all_samples)} samples. Starting mining...")
    
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