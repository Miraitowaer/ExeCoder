import os
import re
import json
import requests
import subprocess
import tempfile
import random
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ast
import difflib
import traceback

# 添加线程锁
lock = threading.Lock()

# ===================== 配置参数 =====================
# 输入目录：存放 dpo_sample_0.json ~ dpo_sample_4.json 的地方
INPUT_DIR = "/data/private/ExeCoder/data/dpo_mining"
# 所有的采样文件名列表
SAMPLE_FILES = [f"dpo_sample_{i}.json" for i in range(5)]

# 输出文件
OUTPUT_FILE = "/data/private/ExeCoder/data/dpo_pairs_strategy1_on_policy.json"
BAD_CHOSEN_DIR = "/data/private/ExeCoder/data/bad_chosen_codes_v3"

SUPPORTED_LANGS = ["python", "java", "cpp"]

# 大模型API配置 (复用 v2 的配置)
LLM_API_URL = "https://aimpapi.midea.com/t-aigc/f-devops-qwen3-coder-480b-a35b-instruct/v1/chat/completions"
LLM_API_KEY = "msk-4b8773bf749c892f2c9803aa69ef94b8b96e7cf807da78cbfdf8606ed919adef"
LLM_MODEL = "f-devops-qwen3-coder-480b-a35b-instruct"
LLM_TIMEOUT = 60
LLM_RETRY_TIMES = 5

# 执行配置
EXEC_TIMEOUT = 10
TEMP_DIR_PREFIX = "exec_v3_temp"
MATCH_THRESHOLD = 0.85

# 并发配置
CONCURRENT_WORKERS = 10  # 稍微降低并发，因为每个任务要跑 6 次代码 (1 GT + 5 Samples)
MAX_CONCURRENT_LLM_CALLS = 10
llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)

# ===================== 基础工具函数 (复用 v2) =====================

def call_llm_api(messages: List[Dict[str, str]]) -> Optional[str]:
    with llm_semaphore:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        payload = {
            "model": LLM_MODEL, "messages": messages, "temperature": 0.1, "max_tokens": 1000, "stream": False
        }
        for retry in range(LLM_RETRY_TIMES):
            try:
                response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content")
                if content is None: raise ValueError("API content is None")
                return content.strip() if isinstance(content, str) else None
            except Exception as e:
                if retry < LLM_RETRY_TIMES - 1: time.sleep(2 ** retry)
        return None

def extract_code_from_markdown(markdown_text: str) -> str:
    if not markdown_text: return ""
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, markdown_text, re.DOTALL)
    return match.group(1).strip() if match else markdown_text.strip()

def detect_lang(item):
    instruction = (item.get('instruction') or '') + " " + (item.get('prompt') or '')
    full_text = instruction.lower()
    if re.search(r"to\s+python", full_text): return "python"
    if re.search(r"to\s+java", full_text): return "java"
    if re.search(r"to\s+(cpp|c\+\+)", full_text): return "cpp"
    code = item.get('generated_output', '')[:200]
    if "def " in code: return "python"
    if "#include" in code: return "cpp"
    if "public class" in code: return "java"
    return None

def extract_java_main_class(code: str) -> Optional[str]:
    match = re.search(r'public\s+class\s+(\w+)', code)
    if match: return match.group(1)
    match = re.search(r'class\s+(\w+)', code)
    return match.group(1) if match else None

def calculate_line_offset(raw_text):
    if "```" in raw_text:
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            return raw_text[:match.start(1)].count('\n')
    return 0

# ===================== 编译与执行函数 =====================

def check_compilation(code: str, lang: str, tmp_dir: str) -> Tuple[bool, str]:
    error_msg = ""
    file_path = ""
    try:
        if lang == "python":
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=tmp_dir, delete=False) as f:
                f.write(code)
                file_path = f.name
            cmd = ["python", "-m", "py_compile", file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0: return False, result.stderr
            return True, ""
        elif lang == "java":
            class_name = extract_java_main_class(code)
            if not class_name: return False, "No Java class found"
            file_name = f"{class_name}.java"
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f: f.write(code)
            result = subprocess.run(["javac", file_name], cwd=tmp_dir, capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return (result.returncode == 0, result.stderr.strip())
        elif lang == "cpp":
            file_name = "code.cpp"
            exe_name = "code"
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f: f.write(code)
            result = subprocess.run(["g++", file_name, "-o", exe_name, "-std=c++11"], cwd=tmp_dir, capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return (result.returncode == 0, result.stderr.strip())
    except Exception as e:
        return False, str(e)
    finally:
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass
    return False, "Unknown error"

def run_code(code: str, lang: str, tmp_dir: str) -> Tuple[bool, str, str]:
    # 返回: (success, stdout, stderr)
    stdout, stderr = "", ""
    file_path = ""
    exe_path = ""
    
    try:
        if lang == "python":
            file_path = os.path.join(tmp_dir, "code.py")
            with open(file_path, "w", encoding="utf-8") as f: f.write(code)
            result = subprocess.run(["python3", file_path], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return (result.returncode == 0, result.stdout.strip(), result.stderr.strip())

        elif lang == "java":
            class_name = extract_java_main_class(code)
            if not class_name: return False, "", "No Java class found"
            # 编译
            c_ok, c_err = check_compilation(code, lang, tmp_dir)
            if not c_ok: return False, "", f"Compilation Failed: {c_err}"
            # 运行
            result = subprocess.run(["java", class_name], cwd=tmp_dir, capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return (result.returncode == 0, result.stdout.strip(), result.stderr.strip())

        elif lang == "cpp":
            c_ok, c_err = check_compilation(code, lang, tmp_dir)
            if not c_ok: return False, "", f"Compilation Failed: {c_err}"
            exe_path = os.path.join(tmp_dir, "code")
            result = subprocess.run([exe_path], cwd=tmp_dir, capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return (result.returncode == 0, result.stdout.strip(), result.stderr.strip())
            
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)
    return False, "", "Unknown Error"

# ===================== 错误分析辅助函数 (复用 v2) =====================

def normalize_text(text: str) -> str:
    if not text: return ""
    try:
        text = text.replace('"', '\\"')
        text = ast.literal_eval(f'"{text}"')
    except:
        text = text.replace(r'\"', '"').replace(r"\\", "\\").replace(r"\'", "'").replace(r"\n", "\n")
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE) # Rm comments
    text = re.sub(r'\s+', '', text) # Rm whitespace
    return text.lower()

def calculate_similarity(text1: str, text2: str) -> float:
    n1, n2 = normalize_text(text1), normalize_text(text2)
    if not n1 or not n2: return 0.0
    return difflib.SequenceMatcher(None, n1, n2).ratio()

def match_code_lines_to_numbers(error_code_lines: List[str], rejected_code: str) -> List[int]:
    matched = []
    if not rejected_code: return []
    rej_lines = rejected_code.split('\n')
    norm_rej = [normalize_text(l) for l in rej_lines]
    
    for err in error_code_lines:
        if not err.strip(): continue
        norm_err = normalize_text(err)
        best_idx, best_score = -1, 0.0
        for i, nr in enumerate(norm_rej):
            if not nr: continue
            score = difflib.SequenceMatcher(None, norm_err, nr).ratio()
            if score > best_score and score >= MATCH_THRESHOLD:
                best_score = score
                best_idx = i
        if best_idx != -1 and best_idx not in matched:
            matched.append(best_idx)
    return sorted(matched)

def parse_compiler_error_lines(error_msg: str, lang: str) -> List[int]:
    lines = set()
    if lang == "java": matches = re.findall(r':(\d+): error:', error_msg)
    elif lang == "cpp": matches = re.findall(r':(\d+):\d+: error:', error_msg)
    elif lang == "python": matches = re.findall(r'line\s+(\d+)', error_msg)
    else: return []
    for m in matches: lines.add(int(m) - 1)
    return sorted(list(lines))

def get_logic_error_code_lines(chosen_code: str, rejected_code: str, chosen_out: str, rejected_out: str, prompt: str, lang: str) -> Tuple[List[str], List[int]]:
    # 此处省略 Prompt 构建细节，复用 v2 的逻辑
    # 为节省 Token，这里简化调用逻辑，实际运行时请确保 v2 的 get_logic_error_code_lines 完整逻辑被包含
    # ... (Please use the exact implementation from mine_all_errors_v2.py) ...
    # 简易实现：
    system_prompt = f"""你是资深{lang}工程师，专注于识别代码逻辑错误。
核心任务：仅分析并返回rejected代码中导致逻辑错误的核心代码行（逐行列出，保留原始格式），无需解释。
背景：
- chosen是正确代码（真值），rejected是逻辑错误代码（执行输出与chosen不一致）
- 逻辑错误指：代码语法正确、能编译执行，但输出结果与chosen不符

输出要求：
1. 仅返回rejected中的错误代码行，每行代码单独列出，用JSON数组格式包裹；
2. 保留代码原始格式（包括空格、符号），不要修改任何内容；
3. 无明确错误代码行时，直接返回空数组 []；
4. 只返回代码行列表，不添加任何额外文字、注释或说明。

正确代码（chosen）：
{chosen_code}
正确输出：
{chosen_out}

错误代码（rejected）：
{rejected_code}
错误输出：
{rejected_out}

代码意图（参考）：
{prompt}

输出格式示例（正确）：
[
"strcpy(s, \"forgeeksskeegfor\");",
"for (int i = 0; i < strlen(s); i++) insert(i);"
]

输出格式示例（无错误）：
[]
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "严格按要求返回rejected中的逻辑错误代码行列表（仅代码，无其他内容）"}
    ]

    response = call_llm_api(messages)
    if not response:
        print("警告：LLM API调用失败，逻辑错误代码行提取失败")
        return [], []

    # 解析大模型返回的错误代码行列表（严格按格式提取）
    try:
        # 提取JSON数组中的内容（兼容带引号/不带引号、带换行/不带换行）
        json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if not json_match:
            print(f"警告：未找到错误代码行列表，响应内容：{response[:300]}...")
            return [], []
        
        # 尝试直接解析整个 JSON 数组，这能正确处理转义字符
        json_str = json_match.group(0)
        try:
            error_code_lines = json.loads(json_str)
        except json.JSONDecodeError:
            # 解析失败（如 JSON 不标准），回退到正则提取并手动清理
            code_content = json_match.group(1).strip()
            if not code_content:
                return [], []
            
            error_code_lines = []
            for line in code_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 去除前后引号（支持单引号、双引号、转义引号）
                line = re.sub(r'^["\'](.*?)["\']$', r'\1', line)
                line = re.sub(r'^\\["\'](.*?)\\["\']$', r'\1', line)
                # 手动处理常见的转义字符残留，如 \" 变为 "
                line = line.replace(r'\"', '"').replace(r"\'", "'")
                if line:
                    error_code_lines.append(line)
        
        # 关键步骤：逐行匹配获取rejected中的相对行号
        matched_line_nums = match_code_lines_to_numbers(error_code_lines, rejected_code)
        return error_code_lines, matched_line_nums
    except Exception as e:
        print(f"错误：解析逻辑错误代码行失败 - {str(e)}，响应内容：{response[:500]}...")
        return [], []

# ===================== 核心处理逻辑 (策略一) =====================

def process_prompt_group(sample_idx, group_samples, root_tmp_dir, stats, results_list):
    """
    处理同一个 Prompt 的一组样本 (包含 5 个不同的 generated_output)
    """
    # 1. 基础信息提取
    base_sample = group_samples[0]
    instruction = (base_sample.get("instruction") or "").strip()
    input_text = (base_sample.get("input") or "").strip()
    gt_output_raw = (base_sample.get("output") or "").strip()
    prompt = f"{instruction}\n\nInput Code: {input_text}"
    
    # 语言检测
    lang = detect_lang(base_sample)
    if not lang or lang not in SUPPORTED_LANGS:
        with lock: stats["lang_detect_fail"] += 1
        return

    # 提取 GT 代码
    gt_code = extract_code_from_markdown(gt_output_raw)
    if not gt_code:
        with lock: stats["empty_gt"] += 1
        return

    # 2. 运行 Ground Truth (确立 Golden Output)
    with tempfile.TemporaryDirectory(dir=root_tmp_dir, prefix=f"p{sample_idx}_gt_") as tmp_dir:
        gt_success, gt_stdout, gt_stderr = run_code(gt_code, lang, tmp_dir)
    
    if not gt_success:
        # GT 运行失败，该题目无法判定对错，跳过
        # save_bad_chosen_code(gt_code, gt_stderr, instruction, lang, BAD_CHOSEN_DIR) # 可选
        with lock: stats["gt_exec_fail"] += 1
        return
    
    golden_output = gt_stdout.strip()

    # 3. 评估 5 个 Candidate
    pass_candidates = [] # list of code
    fail_candidates = [] # list of (code, raw_text, error_type, error_msg, execution_result_tuple)
    
    # 我们需要临时目录来跑每一个 candidate
    # 为了效率，可以重用一个目录？不行，Java文件名冲突。还是每个单独开吧。
    
    for i, sample in enumerate(group_samples):
        gen_raw = (sample.get("generated_output") or "").strip()
        gen_code = extract_code_from_markdown(gen_raw)
        
        if not gen_code: continue

        with tempfile.TemporaryDirectory(dir=root_tmp_dir, prefix=f"p{sample_idx}_c{i}_") as tmp_dir:
            # A. 编译检查
            c_ok, c_err = check_compilation(gen_code, lang, tmp_dir)
            if not c_ok:
                fail_candidates.append({
                    "code": gen_code, "raw": gen_raw, "type": "compilation_error", "msg": c_err, "res": None
                })
                continue
            
            # B. 运行检查
            # 注意：Java 需要重新编译运行，Python/C++ 可以直接运行
            # run_code 内部包含了编译步骤(for java/cpp if needed)，但 check_compilation 已经做了一次
            # 为了简单，直接调用 run_code，它比较 robust
            r_ok, r_out, r_err = run_code(gen_code, lang, tmp_dir)
            
            if not r_ok:
                fail_candidates.append({
                    "code": gen_code, "raw": gen_raw, "type": "execution_error", "msg": r_err, "res": (r_ok, r_out, r_err)
                })
            else:
                # C. 逻辑检查 (对比 Golden Output)
                # 简单去除首尾空白对比
                if r_out.strip() == golden_output:
                    pass_candidates.append(gen_raw) # 存原始文本，方便直接作为 chosen
                else:
                    fail_candidates.append({
                        "code": gen_code, "raw": gen_raw, "type": "logic_error", 
                        "msg": f"Expected: {golden_output}\nGot: {r_out.strip()}", 
                        "res": (r_ok, r_out, r_err)
                    })

    # 4. 构建 Pair (Strategy 1)
    chosen_sample = None
    rejected_info = None # dict
    is_on_policy = False
    
    if len(pass_candidates) > 0 and len(fail_candidates) > 0:
        # [最优情况] Model vs Model
        chosen_sample = random.choice(pass_candidates)
        rejected_info = random.choice(fail_candidates)
        is_on_policy = True
        with lock: stats["pair_model_model"] += 1
        
    elif len(pass_candidates) == 0 and len(fail_candidates) > 0:
        # [保底情况] GT vs Model
        chosen_sample = gt_output_raw
        rejected_info = random.choice(fail_candidates)
        is_on_policy = False
        with lock: stats["pair_gt_model"] += 1
        
    else:
        # 全对 或 全无法编译且无GT(已在前面过滤) -> 跳过
        with lock: stats["no_valid_pair"] += 1
        return

    # 5. 挖掘错误详情 (Mining)
    # 只需要对选中的 rejected_info 进行详细分析
    final_entry = {
        "prompt": prompt,
        "chosen": extract_code_from_markdown(chosen_sample), # 统一存纯代码
        "rejected": rejected_info["raw"], # 存原始 generated_output (含markdown)，方便后续处理保持一致? 
                                          # 不，v2存的是 rejected_raw。但 preprocess 脚本里又处理了。
                                          # 这里为了配合 train_dpo_focus.py，我们保持 rejected 为 raw text
        "lang": lang,
        "error_type": rejected_info["type"],
        "error_msg": rejected_info["msg"],
        "source_policy": "on_policy" if is_on_policy else "off_policy"
    }
    
    # 计算 error_lines / error_code
    rej_code = rejected_info["code"]
    offset = calculate_line_offset(rejected_info["raw"])
    
    if rejected_info["type"] == "compilation_error":
        rel_lines = parse_compiler_error_lines(rejected_info["msg"], lang)
        final_entry["error_lines"] = [l + offset for l in rel_lines]
        final_entry["error_code"] = None # 编译错误通常不需要内容匹配，或者难以提取
        
    elif rejected_info["type"] == "execution_error":
        # 执行错误暂时不调用 LLM 定位行号，留空或简单的 stderr 分析
        # 如果需要高精度，可以把 locate_execution_error_lines 加回来
        final_entry["error_lines"] = []
        final_entry["error_code"] = None
        
    elif rejected_info["type"] == "logic_error":
        try:
            exec_out = rejected_info["res"][1]
            err_codes, rel_lines = get_logic_error_code_lines(
                final_entry["chosen"], rej_code, golden_output, exec_out, prompt, lang
            )
            final_entry["error_code"] = err_codes
            final_entry["error_lines"] = [l + offset for l in rel_lines]
        except Exception as e:
            print(f"[Logic Error Processing Failed] {e}")
            traceback.print_exc()
            # 即使 LLM 分析失败，也保存这个样本，只是没有具体行号
            final_entry["error_lines"] = []
            final_entry["error_code"] = []

    # 添加到结果集
    with lock:
        results_list.append(final_entry)


def main():
    print(f"Loading samples from {INPUT_DIR}...")
    
    # 1. 读取所有文件
    all_data = []
    for fname in SAMPLE_FILES:
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Error: File {fpath} not found.")
            return
        with open(fpath, 'r') as f:
            all_data.append(json.load(f))
    
    # 检查长度一致性
    assert all(len(d) == len(all_data[0]) for d in all_data), "Files have different lengths!"
    total_prompts = len(all_data[0])
    print(f"Loaded 5 files, each with {total_prompts} samples.")

    # 2. 转换为按 Prompt 分组
    # grouped_data[i] = [sample_from_file0, sample_from_file1, ...]
    grouped_data = []
    for i in range(total_prompts):
        group = [data[i] for data in all_data]
        grouped_data.append(group)

    # 3. 并发处理
    stats = {
        "lang_detect_fail": 0, "empty_gt": 0, "gt_exec_fail": 0,
        "pair_model_model": 0, "pair_gt_model": 0, "no_valid_pair": 0
    }
    results = []
    
    os.makedirs(BAD_CHOSEN_DIR, exist_ok=True)
    
    print(f"Starting processing with {CONCURRENT_WORKERS} threads...")
    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as root_tmp_dir:
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = [
                executor.submit(process_prompt_group, i, group, root_tmp_dir, stats, results)
                for i, group in enumerate(grouped_data)
            ]
            
            for future in tqdm(as_completed(futures), total=total_prompts, desc="Mining Pairs"):
                try:
                    future.result() # 这里会抛出线程内的异常
                except Exception as e:
                    print(f"Thread Error: {e}")

    # 4. 保存结果
    print("\n" + "="*50)
    print("Processing Stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved {len(results)} pairs to {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()