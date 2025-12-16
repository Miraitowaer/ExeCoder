import json
import random
from typing import List, Dict

def load_json_data(file_path: str) -> List[Dict]:
    """
    加载JSON数据集文件
    支持两种常见格式：
    1. 完整的JSON数组格式（[{}, {}, ...]）
    2. JSON Lines格式（每行一个JSON对象）
    """
    try:
        # 尝试按JSON数组读取
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        # 按JSON Lines格式读取
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"第{line_num}行JSON格式错误: {e}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    except Exception as e:
        raise Exception(f"读取文件失败: {e}")

def save_json_data(data: List[Dict], file_path: str, json_lines: bool = False):
    """
    保存JSON数据到文件
    :param data: 要保存的数据列表
    :param file_path: 保存路径
    :param json_lines: 是否按JSON Lines格式保存
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if json_lines:
                # JSON Lines格式（每行一个对象）
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            else:
                # 标准JSON数组格式
                json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到: {file_path} (共{len(data)}条)")
    except Exception as e:
        raise Exception(f"保存文件失败: {e}")

def split_dataset(
    input_file: str,
    sft_output: str = "sft_data.json",
    dpo_output: str = "dpo_data.json",
    split_ratio: float = 0.5,
    random_seed: int = 42,
    json_lines: bool = False
):
    """
    分割数据集为SFT和DPO两部分
    :param input_file: 输入JSON文件路径
    :param sft_output: SFT数据集输出路径
    :param dpo_output: DPO数据集输出路径
    :param split_ratio: SFT数据集占比（0-1），这里设为0.5即1:1分割
    :param random_seed: 随机种子（保证可复现）
    :param json_lines: 是否按JSON Lines格式保存输出文件
    """
    # 1. 加载数据
    print("开始加载数据集...")
    data = load_json_data(input_file)
    total_count = len(data)
    if total_count == 0:
        raise ValueError("数据集为空！")
    print(f"成功加载 {total_count} 条数据")

    # 2. 设置随机种子，保证分割结果可复现
    random.seed(random_seed)
    
    # 3. 打乱数据（避免数据分布不均）
    print("打乱数据顺序...")
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 4. 计算分割点
    split_idx = int(total_count * split_ratio)
    
    # 5. 分割数据
    sft_data = shuffled_data[:split_idx]
    dpo_data = shuffled_data[split_idx:]

    # 6. 保存数据
    print("保存SFT数据集...")
    save_json_data(sft_data, sft_output, json_lines)
    
    print("保存DPO数据集...")
    save_json_data(dpo_data, dpo_output, json_lines)

    # 7. 输出分割统计信息
    print("\n分割完成！统计信息：")
    print(f"原始数据总数: {total_count}")
    print(f"SFT数据集数量: {len(sft_data)} ({len(sft_data)/total_count*100:.2f}%)")
    print(f"DPO数据集数量: {len(dpo_data)} ({len(dpo_data)/total_count*100:.2f}%)")

if __name__ == "__main__":
    # ==================== 配置参数 ====================
    INPUT_FILE = "/data/private/ExeCoder/data/XLCoST_data/XLCoST-Instruct/Tuning/code/train.json"  # 替换为你的原始JSON文件路径
    SFT_OUTPUT_FILE = "/data/private/ExeCoder/data/split_sft_data.json"       # SFT数据集输出路径
    DPO_OUTPUT_FILE = "/data/private/ExeCoder/data/split_dpo_data.json"       # DPO数据集输出路径
    RANDOM_SEED = 42                        # 随机种子（可修改）
    JSON_LINES_FORMAT = False               # 是否使用JSON Lines格式保存（根据需求调整）
    # ==================================================

    # 执行分割
    split_dataset(
        input_file=INPUT_FILE,
        sft_output=SFT_OUTPUT_FILE,
        dpo_output=DPO_OUTPUT_FILE,
        split_ratio=0.5,
        random_seed=RANDOM_SEED,
        json_lines=JSON_LINES_FORMAT
    )