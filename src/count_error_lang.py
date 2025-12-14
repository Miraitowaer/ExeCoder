import json
import argparse
from collections import defaultdict

def count_error_by_lang(json_file_path):
    """
    统计JSON数据集中不同错误类型下各语言的样本数量
    :param json_file_path: JSON数据集文件路径
    :return: 统计结果字典
    """
    # 初始化统计字典（默认值为0）
    stats = {
        "compilation_error": defaultdict(int),  # 编译错误
        "execution_error": defaultdict(int),    # 执行错误
        "logic_error": defaultdict(int),        # 逻辑错误
        "unknown_error_type": defaultdict(int), # 未知错误类型
        "unknown_lang": 0,                      # 无lang字段的样本数
        "total": 0                              # 总样本数
    }
    # 支持的语言列表（用于过滤无效值）
    SUPPORTED_LANGS = {"python", "java", "cpp"}

    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到文件 {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误：{json_file_path} 不是合法的JSON文件")
        return None

    # 遍历每个样本统计
    for idx, sample in enumerate(data):
        stats["total"] += 1
        # 提取关键字段
        error_type = sample.get("error_type", "").strip().lower()
        lang = sample.get("lang", "").strip().lower()

        # 处理无lang字段或不支持的语言
        if not lang or lang not in SUPPORTED_LANGS:
            stats["unknown_lang"] += 1
            continue

        # 按错误类型统计
        if error_type == "compilation_error":
            stats["compilation_error"][lang] += 1
        elif error_type == "execution_error":
            stats["execution_error"][lang] += 1
        elif error_type == "logic_error":
            stats["logic_error"][lang] += 1
        else:
            stats["unknown_error_type"][lang] += 1

    return stats

def print_stats_table(stats):
    """
    以表格形式打印统计结果，格式清晰易读
    :param stats: 统计结果字典
    """
    # 定义表头和列名
    error_types = ["编译错误(compilation_error)", "执行错误(execution_error)", "逻辑错误(logic_error)"]
    langs = ["python", "java", "cpp"]
    
    # 打印标题
    print("\n" + "="*80)
    print("📊 样本错误类型&语言分布统计结果")
    print("="*80)
    
    # 打印表头
    header = f"{'错误类型':<30} | {'Python':<10} | {'Java':<10} | {'C++':<10} | {'小计':<10}"
    print(header)
    print("-"*80)
    
    # 打印各错误类型的统计数据
    total_compile = 0
    total_exec = 0
    total_logic = 0
    
    # 编译错误
    compile_py = stats["compilation_error"]["python"]
    compile_java = stats["compilation_error"]["java"]
    compile_cpp = stats["compilation_error"]["cpp"]
    compile_sum = compile_py + compile_java + compile_cpp
    total_compile = compile_sum
    print(f"{'编译错误':<30} | {compile_py:<10} | {compile_java:<10} | {compile_cpp:<10} | {compile_sum:<10}")
    
    # 执行错误
    exec_py = stats["execution_error"]["python"]
    exec_java = stats["execution_error"]["java"]
    exec_cpp = stats["execution_error"]["cpp"]
    exec_sum = exec_py + exec_java + exec_cpp
    total_exec = exec_sum
    print(f"{'执行错误':<30} | {exec_py:<10} | {exec_java:<10} | {exec_cpp:<10} | {exec_sum:<10}")
    
    # 逻辑错误
    logic_py = stats["logic_error"]["python"]
    logic_java = stats["logic_error"]["java"]
    logic_cpp = stats["logic_error"]["cpp"]
    logic_sum = logic_py + logic_java + logic_cpp
    total_logic = logic_sum
    print(f"{'逻辑错误':<30} | {logic_py:<10} | {logic_java:<10} | {logic_cpp:<10} | {logic_sum:<10}")
    
    # 打印分隔线
    print("-"*80)
    
    # 打印总计行
    total_py = compile_py + exec_py + logic_py
    total_java = compile_java + exec_java + logic_java
    total_cpp = compile_cpp + exec_cpp + logic_cpp
    grand_total = total_compile + total_exec + total_logic
    print(f"{'各语言总计':<30} | {total_py:<10} | {total_java:<10} | {total_cpp:<10} | {grand_total:<10}")
    
    # 打印校验信息
    print("\n" + "="*80)
    print("🔍 数据校验 & 异常统计")
    print("="*80)
    print(f"原始统计总数（编译+执行+逻辑）: {total_compile + total_exec + total_logic}")
    print(f"实际遍历总样本数              : {stats['total']}")
    print(f"无有效lang字段的样本数        : {stats['unknown_lang']}")
    print(f"未知错误类型的样本数          : {sum(stats['unknown_error_type'].values())}")
    
    # 验证用户提供的总数
    user_total = 8548 + 5658 + 8449  # 22655
    if grand_total == user_total:
        print(f"✅ 统计总数与预期({user_total})匹配")
    else:
        print(f"❌ 统计总数({grand_total})与预期({user_total})不匹配，请检查数据！")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="统计JSON数据集中不同错误类型下各语言的样本数量")
    parser.add_argument('--file', type=str, required=False, default="/data/private/ExeCoder/data/dpo_pairs_ranked_v4.json", help="JSON数据集文件路径（必填）")
    args = parser.parse_args()

    # 执行统计
    stats = count_error_by_lang(args.file)
    if stats:
        # 打印统计结果
        print_stats_table(stats)

if __name__ == "__main__":
    main()