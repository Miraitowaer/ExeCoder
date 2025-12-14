import os
import re
import json
import subprocess
import tempfile
import random
import threading
import time
import ast
import difflib
import shutil
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 配置参数 =====================
# 输入目录：存放 dpo_sample_0.json ~ dpo_sample_4.json 的地方
INPUT_DIR = "/data/private/ExeCoder/data/dpo_mining"
# 所有的采样文件名列表
SAMPLE_FILES = [f"dpo_sample_{i}.json" for i in range(5)]

# 输出文件
OUTPUT_FILE = "/data/private/ExeCoder/data/dpo_pairs_ranked_v4.json"

SUPPORTED_LANGS = ["python", "java", "cpp"]

# 执行配置
EXEC_TIMEOUT = 5  # 执行超时时间(秒)
TEMP_DIR_PREFIX = "exec_v5_rank_"

# 并发配置
CONCURRENT_WORKERS = 16 
lock = threading.Lock()

# ===================== 基础工具函数 =====================

def extract_code_from_markdown(markdown_text: str) -> str:
    """提取 Markdown 代码块"""
    if not markdown_text: return ""
    # 优先匹配指定语言的代码块
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown_text.strip()

def detect_lang(item):
    """简单检测语言"""
    instruction = (item.get('instruction') or '') + " " + (item.get('prompt') or '')
    full_text = instruction.lower()
    if re.search(r"to\s+python", full_text): return "python"
    if re.search(r"to\s+java", full_text): return "java"
    if re.search(r"to\s+(cpp|c\+\+)", full_text): return "cpp"
    
    # 兜底检测
    code = item.get('generated_output', '')[:200]
    if "def " in code: return "python"
    if "#include" in code: return "cpp"
    if "public class" in code: return "java"
    return None

def extract_java_main_class(code: str) -> Optional[str]:
    """提取 Java 类名"""
    match = re.search(r'public\s+class\s+(\w+)', code)
    if match: return match.group(1)
    match = re.search(r'class\s+(\w+)', code)
    return match.group(1) if match else "Main"

def normalize_text(text: str) -> str:
    """标准化代码文本，用于计算相似度"""
    if not text: return ""
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_similarity(code1: str, code2: str) -> float:
    """计算两段代码的相似度 (0.0 - 1.0)"""
    if not code1 or not code2: return 0.0
    return difflib.SequenceMatcher(None, normalize_text(code1), normalize_text(code2)).ratio()

# ===================== 编译与执行函数 =====================

def run_code(code: str, lang: str, tmp_dir: str) -> Tuple[str, str, str]:
    """
    编译并运行代码
    Returns: (status, stdout, stderr)
    status: 'success', 'compilation_error', 'runtime_error', 'timeout'
    """
    stdout, stderr = "", ""
    try:
        if lang == "python":
            file_path = os.path.join(tmp_dir, "solution.py")
            with open(file_path, "w", encoding="utf-8") as f: f.write(code)
            # 检查语法
            if subprocess.run(["python3", "-m", "py_compile", file_path], capture_output=True).returncode != 0:
                return "compilation_error", "", "Syntax Error"
            # 运行
            result = subprocess.run(["python3", file_path], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return ("success" if result.returncode == 0 else "runtime_error", result.stdout.strip(), result.stderr.strip())

        elif lang == "java":
            class_name = extract_java_main_class(code)
            file_path = os.path.join(tmp_dir, f"{class_name}.java")
            with open(file_path, "w", encoding="utf-8") as f: f.write(code)
            # 编译
            compile_res = subprocess.run(["javac", file_path], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            if compile_res.returncode != 0:
                return "compilation_error", "", compile_res.stderr.strip()
            # 运行
            run_res = subprocess.run(["java", "-cp", tmp_dir, class_name], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return ("success" if run_res.returncode == 0 else "runtime_error", run_res.stdout.strip(), run_res.stderr.strip())

        elif lang == "cpp":
            file_path = os.path.join(tmp_dir, "solution.cpp")
            exe_path = os.path.join(tmp_dir, "solution")
            with open(file_path, "w", encoding="utf-8") as f: f.write(code)
            # 编译
            compile_res = subprocess.run(["g++", file_path, "-o", exe_path, "-std=c++17"], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            if compile_res.returncode != 0:
                return "compilation_error", "", compile_res.stderr.strip()
            # 运行
            run_res = subprocess.run([exe_path], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
            return ("success" if run_res.returncode == 0 else "runtime_error", run_res.stdout.strip(), run_res.stderr.strip())
            
    except subprocess.TimeoutExpired:
        return "timeout", "", "Execution Timed Out"
    except Exception as e:
        return "runtime_error", "", str(e)
    
    return "runtime_error", "", "Unknown Error"

# ===================== 核心处理逻辑 =====================

def process_single_task(task_idx: int, task_group: List[Dict], root_tmp_dir: str, stats: Dict, results_list: List):
    """
    处理单个任务（包含GT和5个采样）
    """
    base_item = task_group[0]
    prompt = (base_item.get("instruction") or "") + "\n" + (base_item.get("input") or "")
    gt_output_raw = base_item.get("output") or ""
    gt_code = extract_code_from_markdown(gt_output_raw)
    
    lang = detect_lang(base_item)
    if not lang or lang not in SUPPORTED_LANGS:
        return 

    # 1. GT 预审 (Ground Truth Sanity Check)
    task_tmp_dir = os.path.join(root_tmp_dir, f"task_{task_idx}")
    os.makedirs(task_tmp_dir, exist_ok=True)
    
    gt_status, gt_stdout, _ = run_code(gt_code, lang, task_tmp_dir)
    shutil.rmtree(task_tmp_dir) 

    if gt_status != "success":
        with lock: stats["skipped_bad_gt"] += 1
        return # GT都跑不通，丢弃
    
    golden_output = gt_stdout.strip()

    # 2. 评估所有采样代码
    candidates = [] 
    
    for i, item in enumerate(task_group):
        gen_raw = item.get("generated_output", "")
        gen_code = extract_code_from_markdown(gen_raw)
        if not gen_code: continue

        sample_tmp_dir = os.path.join(root_tmp_dir, f"task_{task_idx}_s{i}")
        os.makedirs(sample_tmp_dir, exist_ok=True)

        status, stdout, stderr = run_code(gen_code, lang, sample_tmp_dir)
        shutil.rmtree(sample_tmp_dir)

        is_correct = False
        error_type = status
        
        if status == "success":
            if stdout.strip() == golden_output:
                is_correct = True
                error_type = "none"
            else:
                error_type = "logic_error"
        
        sim_to_gt = calculate_similarity(gen_code, gt_code)

        candidates.append({
            "raw_text": gen_raw,
            "code": gen_code,
            "is_correct": is_correct,
            "error_type": error_type,
            "sim_to_gt": sim_to_gt,
            "stderr": stderr,
            "stdout": stdout.strip() # 保存 stdout 用于对比
        })

    # 3. 排序与配对 (Ranking & Pairing)
    
    # --- 确定 Chosen ---
    final_chosen_code = None
    chosen_source = ""
    passed_candidates = [c for c in candidates if c["is_correct"]]
    
    if passed_candidates:
        passed_candidates.sort(key=lambda x: x["sim_to_gt"], reverse=True)
        best_candidate = passed_candidates[0]
        final_chosen_code = best_candidate["code"]
        chosen_source = "model_pass"
    else:
        final_chosen_code = gt_code
        chosen_source = "ground_truth"

    # --- 确定 Rejected ---
    failed_candidates = [c for c in candidates if not c["is_correct"]]
    
    if not failed_candidates:
        with lock: stats["skipped_all_pass"] += 1
        return 

    for c in failed_candidates:
        c["sim_to_chosen"] = calculate_similarity(c["code"], final_chosen_code)

    # 选最像 Chosen 但错了的 (Hard Negative)
    failed_candidates.sort(key=lambda x: x["sim_to_chosen"], reverse=True)
    best_rejected = failed_candidates[0]

    # --- 构建 Error Msg ---
    # 修改点：根据错误类型定制 error_msg
    if best_rejected["error_type"] == "logic_error":
        # 逻辑错误：展示 期望值 vs 实际值
        final_error_msg = f"Expected Output:\n{golden_output}\n\nActual Output:\n{best_rejected['stdout']}"
    else:
        # 编译或运行时错误：展示 stderr
        final_error_msg = best_rejected["stderr"]
        # 如果 stderr 为空但状态是 error (极少数情况)，给个默认提示
        if not final_error_msg:
            final_error_msg = f"Unknown Error ({best_rejected['error_type']})"

    # 截断过长的错误信息，防止 token 爆炸
    if len(final_error_msg) > 2000:
        final_error_msg = final_error_msg[:2000] + "\n... (truncated)"

    # 4. 组装数据
    dpo_entry = {
        "task_id": task_idx,
        "prompt": prompt,
        "chosen": final_chosen_code,
        "rejected": best_rejected["code"],
        "chosen_source": chosen_source,
        "rejected_error_type": best_rejected["error_type"],
        "rejected_error_msg": final_error_msg, # <--- 这里使用了定制后的 msg
        "similarity_score": best_rejected["sim_to_chosen"], 
        "lang": lang
    }
    
    with lock:
        results_list.append(dpo_entry)
        if chosen_source == "model_pass":
            stats["pairs_model_vs_model"] += 1
        else:
            stats["pairs_gt_vs_model"] += 1

def main():
    print(f"🚀 Starting DPO Pair Mining (v5 - Enhanced Logic Error Msg)...")
    print(f"📂 Loading data from {INPUT_DIR}...")
    
    all_files_data = []
    for fname in SAMPLE_FILES:
        fpath = os.path.join(INPUT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                all_files_data.append(json.load(f))
        else:
            print(f"⚠️ Warning: File {fname} not found, skipping.")

    if not all_files_data:
        print("❌ No data loaded.")
        return

    total_tasks = len(all_files_data[0])
    print(f"✅ Loaded {len(all_files_data)} sample files. Total tasks: {total_tasks}")

    tasks_groups = []
    for i in range(total_tasks):
        group = []
        for file_data in all_files_data:
            if i < len(file_data):
                group.append(file_data[i])
        tasks_groups.append(group)

    stats = {
        "skipped_bad_gt": 0,
        "skipped_all_pass": 0,
        "pairs_model_vs_model": 0,
        "pairs_gt_vs_model": 0
    }
    results = []

    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as root_tmp_dir:
        print(f"⚙️  Processing with {CONCURRENT_WORKERS} threads at {root_tmp_dir}...")
        
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = [
                executor.submit(process_single_task, i, group, root_tmp_dir, stats, results)
                for i, group in enumerate(tasks_groups)
            ]
            
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Mining"):
                pass

    print("\n" + "="*40)
    print("📊 Mining Statistics:")
    print(f"   - Total Output Pairs: {len(results)}")
    print(f"   - Pairs (Model vs Model): {stats['pairs_model_vs_model']}")
    print(f"   - Pairs (GT vs Model): {stats['pairs_gt_vs_model']}")
    print(f"   - Skipped (GT Failed): {stats['skipped_bad_gt']}")
    print(f"   - Skipped (All Candidates Passed): {stats['skipped_all_pass']}")
    print("="*40)

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()