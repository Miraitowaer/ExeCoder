import json
from transformers import AutoTokenizer

# 1. 模拟一条数据
sample_data = {
    "rejected": """```java\nimport java . util . * ;\nclass GFG {\n  static String solution ( String s ) {\n    Vector < Character > vowel = new Vector < Character > ( ) ;\n    Vector < Character > consonant = new Vector < Character > ( ) ;\n    for ( char i : s . toCharArray ( ) ) {\n      if ( i =='a'i =='e'i =='i'i =='o'i =='u') {\n        vowel . add ( i ) ;\n      }\n      else {\n        consonant . add ( i ) ;\n      }\n    }\n    int Nc , Nv ;\n    Nv = vowel . size ( ) ;\n    Nc = consonant . size ( ) ;\n    int M = Nc / ( Nv - 1 ) ;\n    String ans = \"\" ;\n    int consotnant_till = 0 ;\n    for ( char i : vowel ) {\n      ans += i ;\n      int temp = 0 ;\n      for ( int j = consotnant_till ;\n      j < Math . min ( Nc , consotnant_till + M ) ;\n      j ++ ) {\n        ans += consonant . get ( j ) ;\n        temp ++ ;\n      }\n      consotnant_till += temp ;\n    }\n    return ans ;\n  }\n  public static void main ( String [ ] args ) {\n    String str = \"aaaabbbcc\" ;\n    System . out . println ( solution ( str ) ) ;\n  }\n}\n\n```""",
    "error_lines": [7,10]  # 假设第 3 行 (index 2) 是错的: print(...)
}

# 2. 加载 Tokenizer (用你现在的模型路径)
model_path = "/data/private/ExeCoder/results/Deepseek-coder-6.7b-instruct-code/checkpoint-400" 
# 如果路径不对，换成 "deepseek-ai/deepseek-coder-6.7b-instruct" 测试
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)

code = sample_data["rejected"]
target_line_indices = set(sample_data["error_lines"])

# 3. 关键步骤：计算每一行的字符范围 (Start, End)
lines = code.split('\n')
line_offsets = []
current_pos = 0
for line in lines:
    # +1 是因为换行符 \n
    line_offsets.append((current_pos, current_pos + len(line))) 
    current_pos += len(line) + 1

print(f"--- Original Code (Error on line {target_line_indices}) ---")
for i, line in enumerate(lines):
    prefix = ">>" if i in target_line_indices else "  "
    print(f"{prefix} Line {i}: {line}")

# 4. Tokenize 并获取 Offset Mapping
# return_offsets_mapping=True 是核心
encoding = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
input_ids = encoding.input_ids
offsets = encoding.offset_mapping

# 5. 映射逻辑：找到属于错误行的 Token
error_token_indices = []

for token_idx, (start_char, end_char) in enumerate(offsets):
    # 判断这个 Token 的字符范围是否落在错误行的字符范围内
    # 简单的判断方法：Token 的中间点落在行内
    token_mid = (start_char + end_char) / 2
    
    is_error_token = False
    for line_idx in target_line_indices:
        line_start, line_end = line_offsets[line_idx]
        if line_start <= token_mid < line_end:
            is_error_token = True
            break
    
    if is_error_token:
        error_token_indices.append(token_idx)

# 6. 验证结果
print(f"\n--- Token Analysis ---")
print(f"Total Tokens: {len(input_ids)}")
print(f"Error Token Indices: {error_token_indices}")

print("\n--- Visual Check ---")
for idx in error_token_indices:
    token_str = tokenizer.decode([input_ids[idx]])
    print(f"Token [{idx}]: '{token_str}' (Should be part of the error line)")