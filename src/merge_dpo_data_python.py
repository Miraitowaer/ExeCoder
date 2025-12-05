import json
import shutil
from datetime import datetime

def process_dpo_json(original_file: str, python_file: str):
    """
    处理DPO JSON文件：
    1. 从original_file中删除lang=python且error_type=execution_error的样本
    2. 将python_file中的内容追加到original_file中
    3. 备份原始文件，输出处理统计信息
    """
    # ========== 1. 备份原始文件（防止数据丢失） ==========
    backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{original_file}.backup_{backup_suffix}"
    try:
        shutil.copy2(original_file, backup_file)
        print(f"✅ 已备份原始文件到：{backup_file}")
    except FileNotFoundError:
        print(f"⚠️  原始文件 {original_file} 不存在，跳过备份")
    except Exception as e:
        print(f"❌ 备份失败：{e}")
        return

    # ========== 2. 读取并处理原始文件 ==========
    try:
        # 读取原始文件
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        if not isinstance(original_data, list):
            print(f"❌ 原始文件 {original_file} 格式错误（非列表）")
            return
        original_count = len(original_data)
        print(f"\n📄 原始文件 {original_file} 样本总数：{original_count}")

        # 过滤：删除lang=python且error_type=execution_error的样本
        filtered_data = []
        deleted_count = 0
        for sample in original_data:
            lang = sample.get("lang", "").strip().lower()
            error_type = sample.get("error_type", "").strip().lower()
            if lang == "python" and error_type == "execution_error":
                deleted_count += 1
                continue
            filtered_data.append(sample)
        
        print(f"🗑️  删除 lang=python 且 error_type=execution_error 的样本数：{deleted_count}")
        print(f"🔍 过滤后剩余样本数：{len(filtered_data)}")

    except FileNotFoundError:
        print(f"❌ 原始文件 {original_file} 不存在，初始化空列表")
        filtered_data = []
    except json.JSONDecodeError:
        print(f"❌ 原始文件 {original_file} 不是合法的JSON文件")
        return
    except Exception as e:
        print(f"❌ 处理原始文件失败：{e}")
        return

    # ========== 3. 读取python专项文件 ==========
    try:
        with open(python_file, 'r', encoding='utf-8') as f:
            python_data = json.load(f)
        if not isinstance(python_data, list):
            print(f"❌ {python_file} 格式错误（非列表），跳过合并")
            python_data = []
        python_count = len(python_data)
        print(f"\n📄 {python_file} 样本总数：{python_count}")

    except FileNotFoundError:
        print(f"❌ {python_file} 不存在，跳过合并")
        python_data = []
        python_count = 0
    except json.JSONDecodeError:
        print(f"❌ {python_file} 不是合法的JSON文件，跳过合并")
        python_data = []
        python_count = 0
    except Exception as e:
        print(f"❌ 读取{python_file}失败：{e}")
        python_data = []
        python_count = 0

    # ========== 4. 合并并保存 ==========
    final_data = filtered_data + python_data
    final_count = len(final_data)
    print(f"\n📊 合并后最终样本总数：{final_count}")

    # 保存到原始文件
    try:
        with open(original_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 处理完成！结果已保存到 {original_file}")
    except Exception as e:
        print(f"❌ 保存文件失败：{e}")
        return

    # ========== 5. 输出最终统计 ==========
    print("\n" + "="*60)
    print("📈 最终处理统计")
    print("="*60)
    print(f"原始样本数        ：{original_count if 'original_count' in locals() else 0}")
    print(f"删除样本数        ：{deleted_count if 'deleted_count' in locals() else 0}")
    print(f"过滤后样本数      ：{len(filtered_data)}")
    print(f"新增python样本数  ：{python_count}")
    print(f"最终样本数        ：{final_count}")
    print("="*60)

def main():
    # 定义文件路径（可根据实际路径修改）
    ORIGINAL_FILE = "/data/private/ExeCoder/data/dpo_errors_pairs.json"
    PYTHON_FILE = "/data/private/ExeCoder/data/dpo_errors_pairs_python.json"

    # 执行处理
    process_dpo_json(ORIGINAL_FILE, PYTHON_FILE)

if __name__ == "__main__":
    main()