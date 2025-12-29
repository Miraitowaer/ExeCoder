import os
import json
import argparse
from typing import Dict, List, Union

def load_json_file(file_path: str) -> Union[List, Dict, None]:
    """加载JSON文件，处理文件不存在/解析错误"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件 {file_path} 不存在")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 仅支持数组或对象类型的JSON
        if not isinstance(data, (list, dict)):
            print(f"❌ 错误：{file_path} 不是数组/对象类型的JSON")
            return None
        return data
    except json.JSONDecodeError as e:
        print(f"❌ 错误：{file_path} JSON解析失败 - {str(e)}")
        return None
    except Exception as e:
        print(f"❌ 错误：读取 {file_path} 失败 - {str(e)}")
        return None

def merge_json_data(
    data1: Union[List, Dict], 
    data2: Union[List, Dict],
    dedup: bool = False,
    dedup_key: str = None
) -> Union[List, Dict]:
    """
    合并两个JSON数据
    :param data1: 第一个JSON数据（数组/对象）
    :param data2: 第二个JSON数据（数组/对象）
    :param dedup: 是否去重（仅数组有效）
    :param dedup_key: 去重依据的字段（如id/prompt，仅数组+dedup=True时有效）
    :return: 合并后的数据
    """
    # 情况1：两个都是数组（最常见，比如之前的badcase输出）
    if isinstance(data1, list) and isinstance(data2, list):
        merged = data1.copy()
        merged.extend(data2)
        
        # 去重逻辑（数组+指定去重字段）
        if dedup and dedup_key:
            seen = set()
            unique_merged = []
            for item in merged:
                if isinstance(item, dict) and dedup_key in item:
                    key_value = item[dedup_key]
                    if key_value not in seen:
                        seen.add(key_value)
                        unique_merged.append(item)
                else:
                    # 无指定字段的项直接保留
                    unique_merged.append(item)
            merged = unique_merged
        return merged
    
    # 情况2：两个都是对象
    elif isinstance(data1, dict) and isinstance(data2, dict):
        merged = data1.copy()
        # 键冲突时，data2覆盖data1
        merged.update(data2)
        return merged
    
    # 情况3：类型不匹配（无法合并）
    else:
        print(f"⚠️ 警告：两个JSON文件类型不匹配（一个是数组，一个是对象），无法合并")
        return None

def main():
    parser = argparse.ArgumentParser(description="合并两个JSON文件（支持数组/对象类型，可选去重）")
    parser.add_argument("--file1", type=str, required=False, default="/data/private/ExeCoder/data/dpo_pairs_ranked_v4.json", help="第一个JSON文件路径（必填）")
    parser.add_argument("--file2", type=str, required=False, default="/data/private/ExeCoder/badcase/Deepseek-coder-6.7b-instruct-code-online-mask/badcases_with_prompt.json", help="第二个JSON文件路径（必填）")
    parser.add_argument("--output", type=str, default="/data/private/ExeCoder/merged_result.json", help="合并后的输出文件路径（默认：merged_result.json）")
    parser.add_argument("--dedup", action="store_true", help="是否对数组型JSON去重（默认关闭）")
    parser.add_argument("--dedup-key", type=str, default="prompt", help="数组去重的依据字段（默认：prompt，可选id/chosen/rejected等）")
    
    args = parser.parse_args()

    # 加载两个JSON文件
    print(f"📂 加载第一个JSON文件：{args.file1}")
    data1 = load_json_file(args.file1)
    print(f"📂 加载第二个JSON文件：{args.file2}")
    data2 = load_json_file(args.file2)
    
    if not data1 or not data2:
        print("❌ 合并失败：至少一个JSON文件加载失败")
        return

    # 合并数据
    print(f"🔗 开始合并JSON数据（去重：{args.dedup}，去重字段：{args.dedup_key}）")
    merged_data = merge_json_data(data1, data2, args.dedup, args.dedup_key)
    if not merged_data:
        return

    # 保存合并结果
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 合并完成！输出文件：{os.path.abspath(args.output)}")
        
        # 输出统计信息
        len1 = len(data1) if isinstance(data1, list) else len(data1.keys())
        len2 = len(data2) if isinstance(data2, list) else len(data2.keys())
        len_merged = len(merged_data) if isinstance(merged_data, list) else len(merged_data.keys())
        print(f"📊 统计：文件1条目数={len1}，文件2条目数={len2}，合并后条目数={len_merged}")
        
    except Exception as e:
        print(f"❌ 保存合并文件失败：{str(e)}")

if __name__ == "__main__":
    main()