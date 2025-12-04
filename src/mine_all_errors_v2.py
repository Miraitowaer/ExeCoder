import os
import re
import json
import requests
import subprocess
import tempfile
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback
import ast

# 添加线程锁，确保统计计数和文件操作的线程安全
lock = threading.Lock()

# ===================== 配置参数（用户需替换）=====================
# 输入配置：目录+正则匹配多文件
INPUT_DIR = "/data/private/ExeCoder/data/dpo_mining"  # 输入文件所在目录（已填用户路径）
INPUT_FILE_PATTERN = r"dpo_sample_.*_debug\.json"  # 匹配 "dpo_sample_*_debug.json"

# 输出文件路径（后缀改为json）
OUTPUT_FILE = "/data/private/ExeCoder/data/dpo_errors_pairs.json"  # 输出带错误信息的DPO样本（JSON数组）
# 新增：错误的Chosen代码保存目录
BAD_CHOSEN_DIR = "/data/private/ExeCoder/data/bad_chosen_codes"

# 支持的编程语言（可扩展）
SUPPORTED_LANGS = ["python", "java", "cpp"]

# 大模型API配置（符合OpenAI协议）
LLM_API_URL = "https://aimpapi.midea.com/t-aigc/f-devops-qwen3-coder-480b-a35b-instruct/v1/chat/completions"
LLM_API_KEY = "msk-4b8773bf749c892f2c9803aa69ef94b8b96e7cf807da78cbfdf8606ed919adef"
LLM_MODEL = "f-devops-qwen3-coder-480b-a35b-instruct"
# 调整：增加超时时间和重试次数
LLM_TIMEOUT = 60  # API请求超时时间（秒）从30延长到60
LLM_RETRY_TIMES = 5  # API调用失败重试次数从3增加到5

# 执行配置
EXEC_TIMEOUT = 10  # 代码执行超时时间（秒）
TEMP_DIR_PREFIX = "code_exec_temp"  # 临时文件前缀

# 文本匹配阈值（避免空格/注释差异导致匹配失败）
MATCH_THRESHOLD = 0.85  # 适当降低阈值，提高匹配成功率

# 并发配置
CONCURRENT_WORKERS = 55  # 并发工作线程数
MAX_CONCURRENT_LLM_CALLS = 55  # 最大并发LLM API调用数
llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)  # 控制LLM API并发量

# ===================== 工具函数 =====================
def call_llm_api(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    调用符合OpenAI协议的大模型API，处理重试和超时，当content为None时也会触发重试
    """
    with llm_semaphore:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000,
            "stream": False
        }

        for retry in range(LLM_RETRY_TIMES):
            try:
                response = requests.post(
                    LLM_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=LLM_TIMEOUT
                )
                response.raise_for_status()  # 抛出HTTP错误（4xx/5xx）
                result = response.json()
                
                # 关键修改：检查content是否为None
                content = result["choices"][0]["message"].get("content")
                if content is None:
                    raise ValueError("API返回的content字段为None")
                
                # 确保content是字符串后再调用strip()
                return content.strip() if isinstance(content, str) else None
                
            except (requests.exceptions.RequestException, ValueError) as e:
                # 同时捕获网络错误和content为None的情况
                print(f"LLM API调用失败（第{retry+1}次重试）：{str(e)}")
                if retry < LLM_RETRY_TIMES - 1:  # 最后一次重试不等待
                    time.sleep(2 ** retry)  # 指数退避重试
        return None

def get_matching_files(input_dir: str, pattern: str) -> List[str]:
    """
    扫描目录，返回所有符合正则模式的文件路径
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"输入目录不存在或不是目录：{input_dir}")
    
    matching_files = []
    regex = re.compile(pattern)
    for file in input_path.glob("*"):
        if file.is_file() and regex.match(file.name):
            matching_files.append(str(file))
    
    if not matching_files:
        raise Warning(f"未找到符合模式 {pattern} 的文件")
    print(f"找到 {len(matching_files)} 个输入文件：{matching_files}")
    return matching_files

def extract_code_from_markdown(markdown_text: str) -> str:
    """
    从Markdown代码块中提取纯代码（去除 ```lang 和 ``` 包裹）
    """
    # 处理None的情况，避免调用strip()时报错
    if not markdown_text:
        return ""
    # 匹配 ```lang\n代码``` 格式
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 无代码块时直接返回原始文本（去除多余空格）
    return markdown_text.strip()

def detect_lang(item):
    # 处理可能的None值
    instruction = item.get('instruction', '') or ''
    prompt = item.get('prompt', '') or ''
    full_text = (instruction + " " + prompt).lower()
    
    if re.search(r"to\s+python", full_text): return "python"
    if re.search(r"to\s+java", full_text): return "java"
    if re.search(r"to\s+(cpp|c\+\+)", full_text): return "cpp"
    
    code_snippet = item.get('rejected', '')[:200] or ''
    if "def " in code_snippet: return "python"
    if "#include" in code_snippet: return "cpp"
    if "public class" in code_snippet: return "java"
    return None

def parse_compiler_error_lines(error_msg: str, lang: str) -> List[int]:
    """
    解析编译器错误信息，返回代码块内的相对错误行号（0基）
    """
    lines = set()
    if not error_msg:
        return []

    # 过滤无法定位到具体行的全局错误
    global_errors = [
        r"cannot find symbol", r"package .* does not exist",
        r"import .* not found", r"no such file or directory",
        r"ModuleNotFoundError", r"ImportError"
    ]
    for pattern in global_errors:
        if re.search(pattern, error_msg, re.IGNORECASE):
            return []

    # 按语言解析行号（编译器输出格式差异）
    if lang == "java":
        matches = re.findall(r':(\d+): error:', error_msg)
    elif lang == "cpp":
        matches = re.findall(r':(\d+):\d+: error:', error_msg)
    elif lang == "python":
        matches = re.findall(r'line\s+(\d+)', error_msg)
    else:
        return []

    # 转换为0基行号（编译器行号从1开始）
    for m in matches:
        lines.add(int(m) - 1)
    return sorted(list(lines))

def calculate_line_offset(raw_text):
    """
    计算从原始文本到提取出的代码之间的行号偏移量。
    逻辑必须与 mine_compilation_errors.py 中的 extract_code 保持一致。
    """
    if "```" in raw_text:
        # 匹配 ```语言名(可选) \n 代码 ```
        # 注意：这里我们需要找到代码块内容的起始位置
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            # match.start(1) 是代码内容(group 1)在原始字符串中的起始索引
            code_start_index = match.start(1)
            
            # 截取代码开始前的所有文本
            prefix_text = raw_text[:code_start_index]
            
            # 计算前缀中的换行符数量，这就是偏移量
            # 例如：前缀有 3 个 \n，说明代码从第 3 行（0-indexed）即第 4 行开始
            return prefix_text.count('\n')
            
    return 0

def check_compilation(code: str, lang: str, tmp_dir: str) -> Tuple[bool, str]:
    error_msg = ""
    file_path = ""
    try:
        if lang == "python":
            return (True, "")
        elif lang == "java":
            # 提取主类名（关键修改）
            class_name = extract_java_main_class(code)
            if not class_name:
                return (False, "未找到有效的 Java 类定义")
            # 文件名必须与类名一致（Java 要求）
            file_name = f"{class_name}.java"
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            # 编译 Java 代码
            result = subprocess.run(
                ["javac", file_name],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT
            )
            error_msg = result.stderr.strip()
            return (result.returncode == 0, error_msg)
        elif lang == "cpp":
            file_name = "code.cpp"
            exe_name = "code.exe" if os.name == "nt" else "code"
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            # 编译C++代码（g++）
            result = subprocess.run(
                ["g++", file_name, "-o", exe_name, "-std=c++11"],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT
            )
            error_msg = result.stderr.strip()
            return (result.returncode == 0, error_msg)
        else:
            return (False, f"不支持的语言：{lang}")
    except subprocess.TimeoutExpired:
        return (False, "编译超时")
    except Exception as e:
        return (False, f"编译异常：{str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

def extract_java_main_class(code: str) -> Optional[str]:
    """提取 Java 代码中的 public 主类名（必须与文件名一致）"""
    # 匹配 public class XXX 格式（Java 要求 public 类名与文件名一致）
    match = re.search(r'public\s+class\s+(\w+)', code)
    if match:
        return match.group(1)
    # 匹配非 public 的 class XXX（如果存在）
    match = re.search(r'class\s+(\w+)', code)
    return match.group(1) if match else None

def run_code(code: str, lang: str, tmp_dir: str) -> Tuple[bool, str, str]:
    """
    运行完整代码（带main函数），返回执行结果（参考transcoder_eval.py执行逻辑）
    返回：(是否成功, stdout, stderr)
    """
    stdout = ""
    stderr = ""
    success = False
    file_paths = []  # 记录临时文件，用于后续清理

    try:
        if lang == "python":
            file_name = "code.py"
            file_path = os.path.join(tmp_dir, file_name)
            file_paths.append(file_path)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            # 运行Python代码
            result = subprocess.run(
                ["python3", file_path],
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            success = result.returncode == 0

        elif lang == "java":
            # 提取主类名（关键修改）
            class_name = extract_java_main_class(code)
            if not class_name:
                return (False, "", "未找到有效的 Java 主类")
            # 先编译（兜底检测）
            compile_ok, compile_err = check_compilation(code, lang, tmp_dir)
            if not compile_ok:
                return (False, "", f"Java 代码编译失败：{compile_err}")
            # 运行编译后的类（使用提取的类名）
            result = subprocess.run(
                ["java", class_name],  # 不再硬编码 Code
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            success = result.returncode == 0
            # 清理 class 文件
            class_files = [f for f in os.listdir(tmp_dir) if f.endswith(".class")]
            for cf in class_files:
                os.remove(os.path.join(tmp_dir, cf))

        elif lang == "cpp":
            # 先编译（兜底检测）
            compile_ok, _ = check_compilation(code, lang, tmp_dir)
            if not compile_ok:
                return (False, "", "C++代码编译失败（执行阶段兜底检测）")
            exe_name = "code.exe" if os.name == "nt" else "code"
            exe_path = os.path.join(tmp_dir, exe_name)
            file_paths.append(exe_path)
            # 运行可执行文件
            result = subprocess.run(
                [exe_path],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            success = result.returncode == 0

        else:
            stderr = f"不支持的语言：{lang}"
    except subprocess.TimeoutExpired:
        stderr = "代码执行超时"
    except Exception as e:
        stderr = f"执行异常：{str(e)}"
    finally:
        # 清理临时文件
        for fp in file_paths:
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except:
                    pass

    return (success, stdout, stderr)

def normalize_text(text: str) -> str:
    """
    标准化文本：处理转义字符、去除空格、注释、换行符，用于代码行匹配
    """
    if not text:  # 处理None或空字符串
        return ""
    
    # 关键修改：处理转义字符（将转义序列转换为实际字符）
    try:
        # 使用ast.literal_eval安全解析字符串，处理大部分转义序列
        # 先包裹成字符串字面量格式（假设文本中没有三重引号冲突）
        text = text.replace('"', '\\"')  # 临时转义双引号，避免解析错误
        parsed = ast.literal_eval(f'"{text}"')  # 解析转义序列
        text = parsed
    except (SyntaxError, ValueError):
        # 解析失败时不处理转义（避免崩溃），仅替换常见转义字符
        text = text.replace(r'\"', '"').replace(r"\\", "\\")
        text = text.replace(r"\'", "'").replace(r"\n", "\n")
    
    # 原有逻辑：去除注释和空白字符
    # 去除单行注释（// 或 #）
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    # 去除多行注释（/* ... */）
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # 去除所有空白字符（空格、制表符、换行符）
    text = re.sub(r'\s+', '', text)
    return text.lower()

def calculate_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度（基于标准化后的字符串匹配）
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    if not norm1 or not norm2:
        return 0.0
    # 计算最长公共子串占比（取较小字符串长度为分母）
    min_len = min(len(norm1), len(norm2))
    max_common = 0
    for i in range(len(norm1)):
        for j in range(len(norm2)):
            k = 0
            while i + k < len(norm1) and j + k < len(norm2) and norm1[i + k] == norm2[j + k]:
                k += 1
            if k > max_common:
                max_common = k
    return max_common / min_len if min_len > 0 else 0.0

def match_code_lines_to_numbers(error_code_lines: List[str], rejected_code: str) -> List[int]:
    """
    核心功能：将大模型返回的逻辑错误代码行列表，逐行匹配到rejected代码中的相对行号（0基）
    支持模糊匹配（忽略空格、注释差异）
    """
    matched_lines = []
    if not rejected_code:  # 处理空代码的情况
        return matched_lines
        
    rejected_lines = rejected_code.split('\n')  # rejected纯代码按行分割
    normalized_rejected = [normalize_text(line) for line in rejected_lines]

    for error_line in error_code_lines:
        if not error_line.strip():
            continue
        normalized_error = normalize_text(error_line)
        if not normalized_error:
            continue

        # 逐行匹配，找相似度最高且满足阈值的行
        best_match_idx = -1
        best_similarity = 0.0
        for idx, norm_rej_line in enumerate(normalized_rejected):
            if not norm_rej_line:
                continue
            similarity = calculate_similarity(normalized_error, norm_rej_line)
            if similarity > best_similarity and similarity >= MATCH_THRESHOLD:
                best_similarity = similarity
                best_match_idx = idx

        if best_match_idx != -1 and best_match_idx not in matched_lines:
            matched_lines.append(best_match_idx)

    return sorted(matched_lines)

# 新增：保存错误的Chosen代码到文件
def save_bad_chosen_code(chosen_code: str, error_msg: str, instruction: str, lang: str, bad_chosen_dir: str):
    """保存执行失败的Chosen代码到指定目录，仅包含代码内容"""
    # 确保目录存在
    os.makedirs(bad_chosen_dir, exist_ok=True)
    
    # 根据语言确定文件扩展名
    ext_map = {
        "python": ".py",
        "java": ".java",
        "cpp": ".cpp"
    }
    ext = ext_map.get(lang, ".txt")  # 默认用txt
    
    # 生成唯一文件名
    timestamp = int(time.time() * 1000)
    hash_str = abs(hash((instruction[:100] + chosen_code[:100]) if (instruction and chosen_code) else str(timestamp))) % 100000
    file_name = f"bad_chosen_{lang}_{timestamp}_{hash_str}{ext}"
    file_path = os.path.join(bad_chosen_dir, file_name)
    
    # 处理None，确保正确输出
    chosen_code = chosen_code or ""
    
    # 仅写入代码内容（保留原始格式）
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(chosen_code)
    
    print(f"已保存错误的Chosen代码到: {file_path}")

# ===================== 错误定位核心函数 =====================
def locate_execution_error_lines(rejected_code: str, error_msg: str, prompt: str, lang: str) -> List[int]:
    """
    调用大模型定位执行错误行（rejected代码中）
    返回：代码块内相对行号（0基）
    """
    system_prompt = f"""你是资深{lang}工程师，擅长定位执行时错误。
任务：根据{lang}代码和执行错误信息，精准找出错误所在的行号（仅返回行号，无额外内容）。
规则：
1. 行号是代码中的实际行号（1基），需转换为0基（减1）；
2. 多个错误行用逗号分隔，无错误行返回空列表；
3. 仅返回行号列表（如 [2,5] 或 []），不添加任何解释。

代码：
{rejected_code}

执行错误信息：
{error_msg}

代码意图（参考）：
{prompt}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "按要求返回执行错误行号列表"}
    ]

    response = call_llm_api(messages)
    if not response:
        return []

    # 解析模型返回的行号（兼容多种格式）
    try:
        line_match = re.search(r'\[?(\d+(?:,\s*\d+)*)\]?', response)
        if not line_match:
            return []
        line_nums = [int(x.strip()) - 1 for x in line_match.group(1).split(',')]
        # 过滤无效行号（超出代码总行数）
        code_lines = rejected_code.split('\n')
        valid_lines = [num for num in line_nums if 0 <= num < len(code_lines)]
        return sorted(list(set(valid_lines)))
    except:
        return []

def compare_outputs(output1: str, output2: str) -> bool:
    """
    格式化对比执行输出（忽略空格、换行符差异）
    """
    def format_output(output: str) -> str:
        # 处理None的情况
        if not output:
            return ""
        # 去除首尾空白、统一换行符、过滤空行
        return '\n'.join([line.strip() for line in output.splitlines() if line.strip()])
    return format_output(output1) == format_output(output2)

def get_logic_error_code_lines(chosen_code: str, rejected_code: str, chosen_out: str, rejected_out: str, prompt: str, lang: str) -> Tuple[List[str], List[int]]:
    """
    核心逻辑：仅调用大模型分析rejected中的逻辑错误代码（逐行返回），再通过逐行匹配获取行数
    返回：(大模型返回的错误代码行列表, 匹配到的rejected相对行号列表)
    """
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
        
        code_content = json_match.group(1).strip()
        if not code_content:
            return [], []
        
        # 按换行分割，处理每行的引号和空白
        error_code_lines = []
        for line in code_content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 去除前后引号（支持单引号、双引号、转义引号）
            line = re.sub(r'^["\'](.*?)["\']$', r'\1', line)
            line = re.sub(r'^\\["\'](.*?)\\["\']$', r'\1', line)
            if line:
                error_code_lines.append(line)
        
        # 关键步骤：逐行匹配获取rejected中的相对行号
        matched_line_nums = match_code_lines_to_numbers(error_code_lines, rejected_code)
        return error_code_lines, matched_line_nums
    except Exception as e:
        print(f"错误：解析逻辑错误代码行失败 - {str(e)}，响应内容：{response[:500]}...")
        return [], []

# ===================== 样本处理函数（用于并发执行） =====================
def process_sample(sample, root_tmp_dir, error_counts, dpo_pairs):
    try:
        with tempfile.TemporaryDirectory(dir=root_tmp_dir, prefix="sample_") as tmp_dir:
            # 修复：使用 or "" 确保变量为字符串后再调用 strip()
            instruction = (sample.get("instruction") or "").strip()
            input_text = (sample.get("input") or "").strip()
            chosen_raw = (sample.get("output") or "").strip()
            rejected_raw = (sample.get("generated_output") or "").strip()

            # 1. 构建prompt：合并instruction和input（任务描述+输入代码）
            prompt = f"{instruction}\n\n输入代码：{input_text}"

            # 2. 提取纯代码（去除Markdown包裹）
            chosen = extract_code_from_markdown(chosen_raw)
            rejected = extract_code_from_markdown(rejected_raw)

            # 3. 自动识别语言（样本无lang字段）
            lang = detect_lang(sample)  # 以正确代码为准识别语言
            if not lang or lang not in SUPPORTED_LANGS:
                lang = detect_lang(rejected) if rejected else None
                if not lang or lang not in SUPPORTED_LANGS:
                    print(f"警告：无法识别语言，跳过样本（instruction前30字：{instruction[:30]}...）")
                    with lock:
                        error_counts["invalid_data"] += 1
                    return
            
            # 跳过无效数据（代码为空）
            if not (chosen and rejected):
                print(f"警告：代码为空，跳过样本（instruction前30字：{instruction[:30]}...）")
                with lock:
                    error_counts["invalid_data"] += 1
                return

            # ===================== 1. 检测编译错误 =====================
            compile_valid, compile_error_msg = check_compilation(rejected, lang, tmp_dir)
            if not compile_valid:
                # 解析错误行并转换为绝对行号
                rel_error_lines = parse_compiler_error_lines(compile_error_msg, lang)
                offset = calculate_line_offset(rejected_raw)  # 基于原始文本计算偏移
                abs_error_lines = [line + offset for line in rel_error_lines]

                result = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected_raw,
                    "lang": lang,
                    "error_type": "compilation_error",
                    "error_msg": compile_error_msg,
                    "error_lines": abs_error_lines,  # 原始文本中的绝对行号
                    "error_code": None  # 编译错误无error_code
                }
                
                with lock:
                    dpo_pairs.append(result)
                    error_counts["compilation_error"] += 1
                return

            # ===================== 2. 检测执行错误 =====================
            exec_valid, exec_stdout, exec_stderr = run_code(rejected, lang, tmp_dir)
            if not exec_valid or exec_stderr:
                execution_error_msg = exec_stderr or "执行失败但未返回错误信息"
                # 调用大模型定位错误行
                rel_error_lines = locate_execution_error_lines(rejected, execution_error_msg, prompt, lang)
                offset = calculate_line_offset(rejected_raw)
                abs_error_lines = [line + offset for line in rel_error_lines]

                result = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected_raw,
                    "lang": lang,
                    "error_type": "execution_error",
                    "error_msg": execution_error_msg,
                    "error_lines": abs_error_lines,
                    "error_code": None  # 执行错误无error_code
                }
                
                with lock:
                    dpo_pairs.append(result)
                    error_counts["execution_error"] += 1
                return

            # ===================== 3. 检测逻辑错误 =====================
            # 运行chosen代码获取正确输出（chosen是真值，理论上无错误）
            chosen_valid, chosen_stdout, chosen_stderr = run_code(chosen, lang, tmp_dir)
            if not chosen_valid or chosen_stderr:
                error_msg = chosen_stderr or "Chosen代码执行失败但未返回错误信息"
                print(f"警告：Chosen代码执行失败（instruction前30字：{instruction[:30]}...），错误：{error_msg[:100]}...")
                
                # 新增：保存错误的Chosen代码
                with lock:
                    save_bad_chosen_code(chosen, error_msg, instruction, lang, BAD_CHOSEN_DIR)
                    error_counts["no_error"] += 1
                return

            # 对比输出是否一致
            if not compare_outputs(chosen_stdout, exec_stdout):
                # 核心流程：大模型提取错误代码行 → 逐行匹配获取行数
                error_code_lines, rel_error_lines = get_logic_error_code_lines(
                    chosen, rejected, chosen_stdout, exec_stdout, prompt, lang
                )
                # 转换为绝对行号
                offset = calculate_line_offset(rejected_raw)
                abs_error_lines = [line + offset for line in rel_error_lines]

                result = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected_raw,
                    "lang": lang,
                    "error_type": "logic_error",
                    "error_msg": (
                        f"Chosen输出：{chosen_stdout}\n"
                        f"Rejected输出：{exec_stdout}"
                    ),
                    "error_code": error_code_lines,  # 大模型返回的逐行错误代码
                    "error_lines": abs_error_lines  # 匹配到的原始文本绝对行号
                }
                
                with lock:
                    dpo_pairs.append(result)
                    error_counts["logic_error"] += 1
            else:
                # 无错误（rejected代码正确，跳过）
                with lock:
                    error_counts["no_error"] += 1

    except Exception as e:
        print(f"处理样本失败：{str(e)}")
        # traceback.print_exc()
        with lock:
            error_counts["invalid_data"] += 1

# ===================== 主逻辑 =====================
def main():
    # 初始化统计
    error_counts = {
        "compilation_error": 0,
        "execution_error": 0,
        "logic_error": 0,
        "no_error": 0,
        "invalid_data": 0,
        "file_read_error": 0,
        "bad_chosen": 0  # 新增：统计错误的Chosen代码数量
    }
    dpo_pairs = []  # 所有样本存入列表，最终输出为JSON数组

    try:
        # 第一步：获取所有符合条件的输入文件
        input_files = get_matching_files(INPUT_DIR, INPUT_FILE_PATTERN)
    except Exception as e:
        print(f"获取输入文件失败：{str(e)}")
        return

    # 第二步：读取所有文件的样本（支持JSON数组格式）
    all_samples = []
    for file in input_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                # 读取整个文件作为JSON数组（解决之前的读取错误）
                data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
                else:
                    # 兼容单个JSON对象的文件
                    all_samples.append(data)
            print(f"成功读取文件 {file}，样本数：{len(data) if isinstance(data, list) else 1}")
        except Exception as e:
            print(f"读取文件 {file} 失败：{str(e)}")
            error_counts["file_read_error"] += 1
            continue
    print(f"\n总样本数：{len(all_samples)}")

    # 第三步：并发处理样本（字段映射+错误检测）
    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as tmp_dir:
        # 使用线程池并发处理样本
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            # 提交所有任务
            futures = [
                executor.submit(
                    process_sample, 
                    sample, 
                    tmp_dir,  # 这里传入的是根临时目录
                    error_counts,
                    dpo_pairs
                ) 
                for sample in all_samples
            ]
            
            # 显示进度
            for _ in tqdm(as_completed(futures), total=len(all_samples), desc="处理样本"):
                pass

    # 第四步：保存为JSON文件
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)  # indent=2便于阅读
        print(f"\n✅ 成功保存JSON文件：{OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 保存文件失败：{str(e)}")
        return

    # 打印统计结果
    print("\n" + "="*50)
    print("样本处理统计：")
    for err_type, count in error_counts.items():
        print(f"{err_type}: {count}")
    print(f"生成DPO样本数：{len(dpo_pairs)}")
    print(f"错误的Chosen代码保存目录：{BAD_CHOSEN_DIR}")
    print(f"输出文件格式：JSON（数组）")
    print(f"输出文件路径：{OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()