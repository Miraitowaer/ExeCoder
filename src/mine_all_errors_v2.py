import os
import sys
import json
import re
import tempfile
import subprocess
import argparse
import glob
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 0. 配置区域 (请在此处填入你的 API Key 和 URL)
# ==============================================================================
API_KEY = "msk-9e80428b8a8e4baa47e44ccb8dc96c4e1e59a80a0f2001b0d6efa63ed7b8ea76"  # 请替换为你的 Key
API_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions" # 假设的基础 URL，根据实际情况调整

# ==============================================================================
# 1. 基础设施配置 (保持不变，用于编译 Runner)
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
# 2. 核心组件：追踪器 (只负责运行，不再负责逻辑Diff)
# ==============================================================================

class Tracer:
    def __init__(self):
        self.tools_jar = find_tools_jar()

    def run(self, code, lang, temp_dir, tag=""):
        """
        运行代码并获取标准输出。
        Returns: (success: bool, output: str)
        """
        if not code or not code.strip(): return False, ""
        try:
            if lang == "cpp": return self._run_cpp(code, temp_dir, tag)
            if lang == "java": return self._run_java(code, temp_dir, tag)
            if lang == "python": return self._run_python(code, temp_dir, tag)
        except Exception:
            return False, ""
        return False, ""

    def _run_cpp(self, code, temp_dir, tag):
        filename = f"prog_{tag}.cpp"
        src = os.path.join(temp_dir, filename)
        exe = os.path.join(temp_dir, f"prog_{tag}")
        with open(src, "w") as f: f.write(code)
        
        try:
            subprocess.run(f"g++ -O2 -w {src} -o {exe}", shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return False, "Compilation Failed" # 这里的失败会被外层 compilation check 捕获

        try:
            res = subprocess.run(exe, cwd=temp_dir, capture_output=True, text=True, timeout=5)
            if res.returncode != 0: return False, res.stderr
            return True, res.stdout
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    def _prepare_java_code(self, code, temp_dir, tag=""):
        # 移除 package 以便直接运行
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
            subprocess.run(f"javac -nowarn {src_path}", shell=True, cwd=temp_dir, check=True, capture_output=True)
            
            cmd = f"java -cp . {cls_name}"
            res = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True, timeout=5)
            if res.returncode != 0: return False, res.stderr
            return True, res.stdout
        except subprocess.CalledProcessError:
            return False, "Compilation Failed"
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    def _run_python(self, code, temp_dir, tag):
        script_name = f"script_{tag}.py"
        script_path = os.path.join(temp_dir, script_name)
        with open(script_path, "w") as f: f.write(code)
        
        try:
            res = subprocess.run([sys.executable, script_name], cwd=temp_dir, capture_output=True, text=True, timeout=5)
            if res.returncode != 0: return False, res.stderr
            return True, res.stdout
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

# ==============================================================================
# 3. 新增组件：DeepSeek LLM 调试器
# ==============================================================================

class DeepSeekDebugger:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.model = "/model/qwen3-235b-a22b"

    def analyze_error(self, chosen_code, rejected_code, error_type, instruction=""):
        """
        调用 DeepSeek-R1 分析错误。
        根据 error_type 构造不同的 Prompt。
        """
        
        system_prompt = "You are an expert code debugger and auditor. Your task is to analyze incorrect code compared to a correct reference."
        
        user_prompt = ""
        if error_type == "execution_error":
            user_prompt = f"""
### Instruction:
{instruction}

### Correct Code (Chosen):
{chosen_code}

### Incorrect Code (Rejected):
{rejected_code}

### Task:
The "Incorrect Code" failed to execute (crashed or timed out).
1. Identify the specific lines or logic causing the runtime crash or infinite loop.
2. Explain the reason for the execution failure.

### Output Format (Strict JSON):
{{
    "error_code": ["<code snippet causing crash>", "<another snippet if needed>"],
    "error_msg": "<concise explanation of the crash>"
}}
"""
        else: # logic_error
            user_prompt = f"""
### Instruction:
{instruction}

### Correct Code (Chosen):
{chosen_code}

### Incorrect Code (Rejected):
{rejected_code}

### Task:
The "Incorrect Code" runs but produces the WRONG output compared to the "Correct Code".
1. Compare the logic of the two codes.
2. Identify the specific lines in "Incorrect Code" that contain the logical flaw.
3. Explain why the logic is incorrect.

### Output Format (Strict JSON):
{{
    "error_code": ["<incorrect code snippet>", "<associated snippet>"],
    "error_msg": "<concise explanation of the logic error>"
}}
"""

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "temperature": 0.1
        }

        try:
            response = requests.post(API_URL, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            
            # 解析 JSON
            return self._clean_and_parse_json(content)
        except Exception as e:
            print(f"[LLM Error] {e}")
            return {"error_code": [], "error_msg": "LLM Analysis Failed"}

    def _clean_and_parse_json(self, text):
        try:
            # 尝试直接解析
            return json.loads(text)
        except:
            # 提取 markdown 代码块中的 json
            match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
            if match:
                try: return json.loads(match.group(1))
                except: pass
            # 简单的正则提取大括号内容
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try: return json.loads(match.group(0))
                except: pass
            
            return {"error_code": [], "error_msg": text.strip()}

# ==============================================================================
# 4. 工作流：ErrorMiner V2
# ==============================================================================

class ErrorMiner:
    def __init__(self):
        self.tracer = Tracer()
        self.debugger = DeepSeekDebugger()

    def detect_lang(self, item):
        # 简单的启发式语言检测
        text = (item.get('instruction', '') + " " + item.get('input', '')).lower()
        match = re.search(r"\bto\s+(c\+\+|cpp|java|python)\b", text)
        if match:
            target = match.group(1)
            if target in ["c++", "cpp"]: return "cpp"
            if target == "java": return "java"
            if target == "python": return "python"
        # Fallback based on code content
        code = item.get('output', '') 
        if not code: return None
        if "public class" in code or "public static void main" in code: return "java"
        if "#include" in code or "using namespace" in code: return "cpp"
        if "def " in code and ":" in code: return "python"
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

        prompt_text = item.get('instruction', '') + "\n" + item.get('input', '')
        chosen = item.get('output', '')
        rejected = item.get('generated_output', '')
        
        chosen_pure = self._extract_code(chosen)
        rejected_pure = self._extract_code(rejected)
        
        if not chosen_pure or not rejected_pure: return None
        
        offset = self._calculate_line_offset(rejected)

        final_sample = {
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
            "lang": lang,
            "error_type": None,
            "error_code": [],
            "error_msg": ""
        }

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                
                # 1. 编译检查 (Compilation Check)
                is_compiled, compile_msg = self._check_compile(rejected_pure, lang, tmp_dir)
                
                if not is_compiled:
                    final_sample['error_type'] = "compilation_error"
                    final_sample['error_msg'] = compile_msg
                    rel_lines = self._parse_compiler_error_lines(compile_msg, lang)
                    final_sample['error_lines'] = [l + offset for l in rel_lines]
                    return final_sample

                # 2. 运行检查 (Execution Check) - Rejected
                success_r, out_r = self.tracer.run(rejected_pure, lang, tmp_dir, "rej")
                
                # 3. 运行检查 (Execution Check) - Chosen (作为真值对照)
                success_c, out_c = self.tracer.run(chosen_pure, lang, tmp_dir, "gold")
                
                if not success_c: 
                    # 连 Chosen 都跑不通，说明样本有问题或者环境缺依赖，跳过
                    return None 

                # ============================================================
                # A. Execution Error (Rejected 运行失败/崩溃)
                # ============================================================
                if not success_r:
                    final_sample["error_type"] = "execution_error"
                    # 调用 LLM 分析崩溃原因
                    print(f"  -> [LLM] Analyzing Execution Error for {lang}...")
                    llm_result = self.debugger.analyze_error(chosen_pure, rejected_pure, "execution_error", prompt_text)
                    final_sample["error_code"] = llm_result.get("error_code", [])
                    final_sample["error_msg"] = llm_result.get("error_msg", "Runtime Error")
                    return final_sample

                # ============================================================
                # B. Logic Error (运行成功但输出不一致)
                # ============================================================
                # 简单的字符串归一化对比 (去掉空白符)
                if out_c.strip().replace(" ", "") != out_r.strip().replace(" ", ""):
                    final_sample["error_type"] = "logic_error"
                    # 调用 LLM 分析逻辑错误
                    print(f"  -> [LLM] Analyzing Logic Error for {lang}...")
                    llm_result = self.debugger.analyze_error(chosen_pure, rejected_pure, "logic_error", prompt_text)
                    final_sample["error_code"] = llm_result.get("error_code", [])
                    final_sample["error_msg"] = llm_result.get("error_msg", "Output Mismatch")
                    return final_sample

                # 如果输出一致，说明是 Correct，忽略
                return None

        except Exception as e:
            print(f"Processing Error: {e}")
            return None

# ==============================================================================
# 5. Main Entry
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./data/dpo_mining")
    parser.add_argument('--output_file', type=str, default="./data/dpo_train_llm_mined.json")
    parser.add_argument('--workers', type=int, default=1)
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
        
    print(f"Loaded {len(all_samples)} samples. Starting mining...")
    
    miner = ErrorMiner()
    results = []
    
    # 使用线程池并行处理
    # 注意：由于要调用 API，建议 worker 数量根据 API Rate Limit 调整，不要太高
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(miner.process, item) for item in all_samples]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                res = fut.result()
                if res: results.append(res)
            except Exception as e:
                pass
            
    print(f"Mining complete. Found {len(results)} valid error samples.")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()