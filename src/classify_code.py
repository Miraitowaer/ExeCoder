import json
import os
import re
import time
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 请替换为你申请到的 DeepSeek API Key 和 URL
API_KEY = "msk-9e80428b8a8e4baa47e44ccb8dc96c4e1e59a80a0f2001b0d6efa63ed7b8ea76"
BASE_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1"  # 或者是你的具体 URL
MODEL_NAME = "/model/qwen3-235b-a22b" # 如果是 R1，通常是 deepseek-reasoner，具体看官方文档

# 输入和输出文件路径
INPUT_FILE = "/data/private/ExeCoder/data/XLCoST_data/XLCoST-Instruct/NL/python.json"   # 你的原始数据文件
OUTPUT_FILE = "/data/private/ExeCoder/junk/dataset_labeled_v1.json" # 结果保存文件

# 定义分类标签（确保模型只从这里面选）
CATEGORIES = [
    "Hash Table", "Two Pointers", "Sliding Window", "Substring", 
    "Array", "Matrix", "Linked List", "Binary Tree", "Graph Theory", 
    "Backtracking", "Binary Search", "Stack", "Heap", 
    "Greedy Algorithm", "Dynamic Programming", "Math", "Bit Manipulation"
]
# ===========================================

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, default_headers={"AIGC-USER": "songzx28"})

def construct_system_prompt():
    """构建系统提示词"""
    categories_str = ", ".join(CATEGORIES)
    prompt = f"""You are an expert algorithm classification system. 
Your task is to analyze the provided "Code" and "Natural Language Description (NL)" and classify the problem into ONE of the following fine-grained algorithmic categories:

{categories_str}

**Requirements:**
1. Analyze the logic, data structures used, and the algorithmic intent.
2. Select the most specific and representative category from the list above.
3. If multiple categories apply, choose the most dominant algorithmic technique (e.g., if DP is used on an Array, output "Dynamic Programming").
4. You must select a category name in the above list and do not generate other category names yourself. Do not output reasoning, explanations, or any other text.
"""
    return prompt

def clean_response(text):
    """
    清理 DeepSeek-R1 可能产生的思维链内容 (<think>...</think>)
    只保留最终的分类标签。
    """
    # 移除 <think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 去除首尾空白
    return text.strip()

def classify_item(item):
    """调用 LLM 进行单条分类"""
    code_content = item.get("Code", "")
    nl_content = item.get("NL", "")
    
    user_prompt = f"Natural Language Description:\n{nl_content}\n\nCode Snippet:\n{code_content}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": construct_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.1, # 降低随机性，使分类更确定
            max_tokens=50,    # 我们只需要一个短标签，不需要太长
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False 
                }
            }
        )
        # print(response)
        raw_content = response.choices[0].message.content
        category = clean_response(raw_content)
        return category
        
    except Exception as e:
        print(f"\nError processing ID {item.get('ID', 'Unknown')}: {e}")
        return None

def main():
    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果数据不是列表格式（比如是一行一个JSON对象的jsonl），需要相应调整读取方式
    if not isinstance(data, list):
        print("错误：JSON文件根节点应该是列表 list []")
        return

    print(f"共加载 {len(data)} 条数据，开始分类...")

    # 2. 遍历处理（带进度条）
    results = []
    # 如果想支持断点续传，可以先读取 output 文件检查已处理的 ID
    
    for item in tqdm(data):
        # 如果已经有分类了，可以跳过（可选）
        # if "Category" in item:
        #     results.append(item)
        #     continue
            
        category = classify_item(item)
        
        if category:
            item["Category"] = category
        else:
            item["Category"] = "Unknown" # 标记失败的
            
        results.append(item)
        
        time.sleep(3)
        
        # 可选：每处理10条保存一次，防止程序崩溃数据丢失
        if len(results) % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # 3. 最终保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n处理完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()