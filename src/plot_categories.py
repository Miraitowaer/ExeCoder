import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import os

# ================= 配置区域 =================
INPUT_FILE = "/data/private/ExeCoder/pic/plot_data/dataset_labeled_v1.json"
OUTPUT_BAR_FILE = "/data/private/ExeCoder/pic/distribution_bar1.png" # PDF 格式适合论文排版 (矢量图)
OUTPUT_PIE_FILE = "/data/private/ExeCoder/pic/distribution_pie1.png"
# ===========================================

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return []

def plot_academic_bar(df):
    """
    生成水平条形图 (修复版：动态高度，防止标签重叠)
    """
    # 1. 设置通用字体
    plt.rcParams['font.family'] = 'sans-serif' # 换成无衬线字体通常更清晰，或者保持 'serif'
    # plt.rcParams['font.sans-serif'] = ['Arial'] # 如果有 Arial 最好，没有就算了
    
    # 2. 排序数据
    df_sorted = df.sort_values('Count', ascending=False)
    
    # 3. 【关键修改】动态计算画布高度
    # 基础高度 2 + 每个条形分配 0.6 英寸的高度
    # 这样如果有 20 个分类，高度就是 14 英寸，绝对不会挤
    dynamic_height = max(6, len(df_sorted) * 0.6) 
    
    plt.figure(figsize=(12, dynamic_height))
    sns.set_style("whitegrid")
    
    # 4. 绘制条形图
    bar_plot = sns.barplot(
        x='Count', 
        y='Category', 
        data=df_sorted, 
        palette="viridis",
        edgecolor="0.2",
        linewidth=1.0 # 边框线宽
    )
    
    # 5. 添加数值标签 (放在条形右侧)
    max_count = df['Count'].max()
    for i, p in enumerate(bar_plot.patches):
        width = p.get_width()
        plt.text(
            width + (max_count * 0.02), # x: 条形末尾 + 缓冲距离
            p.get_y() + p.get_height() / 2, # y: 居中
            f'{int(width)}', 
            va='center', 
            fontsize=12, 
            fontweight='bold',
            color='black'
        )

    # 6. 设置标签和标题
    plt.xlabel('Number of Samples', fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel('Algorithmic Category', fontsize=14, fontweight='bold', labelpad=15)
    plt.title('Distribution of Code Categories', fontsize=16, fontweight='bold', pad=20)
    
    # 7. 调整 Y 轴标签字体大小
    plt.tick_params(axis='y', labelsize=12) # 确保文字足够大且清晰
    plt.tick_params(axis='x', labelsize=12)

    # 8. 保存 (关键：bbox_inches='tight' 确保长标签不被裁掉)
    plt.tight_layout()
    plt.savefig(OUTPUT_BAR_FILE, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ 条形图已保存至: {OUTPUT_BAR_FILE} (高度已自动调整为 {dynamic_height} 英寸)")
    plt.close()

def plot_academic_pie(df):
    """生成环形图 (保持不变，只做微调)"""
    df_sorted = df.sort_values('Count', ascending=False)
    
    # 合并尾部数据，防止饼图太碎
    TOP_N = 8 
    if len(df_sorted) > TOP_N:
        top_df = df_sorted.head(TOP_N)
        others_val = df_sorted.iloc[TOP_N:]['Count'].sum()
        others_df = pd.DataFrame([{'Category': 'Others', 'Count': others_val}])
        plot_df = pd.concat([top_df, others_df], ignore_index=True)
    else:
        plot_df = df_sorted

    plt.figure(figsize=(10, 9))
    
    wedges, texts, autotexts = plt.pie(
        plot_df['Count'], 
        labels=plot_df['Category'], 
        autopct='%1.1f%%', 
        startangle=140,
        colors=sns.color_palette("pastel"),
        pctdistance=0.85, 
        textprops={'fontsize': 12}, # 调大字体
        wedgeprops={'width': 0.4, 'edgecolor': 'w'}
    )
    
    plt.title('Category Proportion', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_PIE_FILE, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ 环形图已保存至: {OUTPUT_PIE_FILE}")
    plt.close()

def main():
    data = load_data()
    if not data: return

    categories = [item.get("Category", "Unknown") for item in data if item.get("Category")]
    if not categories:
        print("⚠️ 数据中没有找到有效的 'Category' 字段。")
        return

    counter = Counter(categories)
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df.columns = ['Category', 'Count']
    
    # 过滤掉极少的类别（可选：例如数量小于总数 1% 的可以忽略或合并）
    # df = df[df['Count'] > 5] 

    print("正在绘图...")
    plot_academic_bar(df)
    plot_academic_pie(df)
    print("🎉 完成！请检查新生成的 _fixed.png 图片。")

if __name__ == "__main__":
    main()