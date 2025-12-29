import json
import re
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher
from pathlib import Path
import matplotlib.font_manager as fm

# 字体设置（兼容无中文字体环境）
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def extract_code_from_markdown(markdown_text: str) -> str:
    """从Markdown中提取纯代码"""
    if not markdown_text:
        return ""
    # 匹配代码块
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown_text.strip()

def calculate_token_overlap(chosen: str, rejected: str) -> float:
    """计算两个代码片段的Token重叠率（基于序列匹配）"""
    # 简单Token化（按空白分割，保留标点）
    chosen_tokens = chosen.split()
    rejected_tokens = rejected.split()
    
    if not chosen_tokens or not rejected_tokens:
        return 0.0  # 空字符串重叠率为0
    
    # 使用difflib计算序列相似度
    matcher = SequenceMatcher(None, chosen_tokens, rejected_tokens)
    return matcher.ratio()  # 返回0-1之间的相似度

def main():
    # 数据集路径
    data_path = Path("/data/private/ExeCoder/data/dpo_pairs_ranked_v4.json")
    
    # 读取数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total_samples = len(data)
    if total_samples == 0:
        print("数据集为空，无法进行统计")
        return
    
    # 初始化重叠率区间统计（0-10%, 10-20%, ..., 90-100%）
    bins = [0.0 + i*0.1 for i in range(11)]  # [0.0, 0.1, ..., 1.0]
    counts = [0] * 10  # 10个区间
    
    # 处理每个样本
    for idx, sample in enumerate(data):
        # 提取代码
        chosen_code = extract_code_from_markdown(sample.get("chosen", ""))
        rejected_code = extract_code_from_markdown(sample.get("rejected", ""))
        
        # 计算重叠率
        overlap = calculate_token_overlap(chosen_code, rejected_code)
        
        # 统计区间
        for i in range(10):
            if bins[i] <= overlap < bins[i+1]:
                counts[i] += 1
                break
        else:
            # 处理1.0（100%）的情况
            counts[-1] += 1
        
        # 进度提示
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{total_samples} 样本")
    
    # 计算每个区间的百分比
    percentages = [(count / total_samples) * 100 for count in counts]
    
    # 生成横轴标签
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(10)]
    
    # 绘制图形
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制条形图（左侧纵轴）
    bars = ax1.bar(labels, counts, color='skyblue', label='样本数量')
    ax1.set_xlabel("Token重叠率区间")
    ax1.set_ylabel("样本数量", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height}', ha='center', va='bottom', fontsize=9)
    
    # 创建右侧纵轴并绘制折线图
    ax2 = ax1.twinx()  # 共享x轴
    ax2.plot(labels, percentages, color='red', marker='o', linestyle='-', linewidth=2, markersize=6, label='占比百分比')
    ax2.set_ylabel("占总样本百分比 (%)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, max(percentages) * 1.2 if percentages else 100)  # 留一些余量
    
    # 添加折线图数值标签
    for i, p in enumerate(percentages):
        ax2.text(i, p + 1, f'{p:.1f}%', ha='center', va='bottom', color='red', fontsize=9)
    
    # 添加标题和网格
    plt.title("Chosen与Rejected的Token重叠率分布（数量与百分比）")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = "/data/private/ExeCoder/pic/token_overlap_distribution_with_percentage.png"
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至 {output_path}")
    plt.show()

if __name__ == "__main__":
    main()