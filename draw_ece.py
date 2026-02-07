import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def safe_float_convert(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# 数据集列表
datasets = [
    'caltech101'
]

# 初始化存储结构
bins = [
    "[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4)", "[0.4, 0.5)",
    "[0.5, 0.6)", "[0.6, 0.7)", "[0.7, 0.8)", "[0.8, 0.9)",
    "[0.9, 1.0)"
]
summary_data = {bin_: {'confidence': [], 'accuracy': [], 'num_samples': []} for bin_ in bins}
ece_data = []
aece_data = []

# 遍历所有数据集
for dataset in datasets:
    base_path = f"/root/autodl-tmp/CLAP-LaplaceGGN/LaplaceGgnOutput/FINAL/debug/caltech101-nonlinear/{dataset}/SGD_lr1e-1_B256_ep300/KL"
    print(os.path.exists(base_path))  # 检查路径是否存在
    # 遍历所有shots和seeds
    shots = ['1shots', '2shots', '4shots', '8shots', '16shots', '32shots']
    seeds = ['1seed', '2seed', '3seed']

    for shot in shots:
        for seed in seeds:
            file_path = os.path.join(base_path, shot, seed, "confidence_coverage_stats.csv")
            print(file_path)
            if os.path.exists(file_path):
                try:
                    # 读取CSV文件，处理可能的空值
                    df = pd.read_csv(file_path)

                    # 确保数值列是浮点数
                    df['accuracy'] = df['accuracy'].apply(safe_float_convert)
                    df['mean_confidence'] = df['mean_confidence'].apply(safe_float_convert)
                    df['num_samples'] = df['num_samples'].apply(safe_float_convert)

                    # 处理每个bin的数据
                    for _, row in df.iterrows():
                        if row['confidence_bin'] in bins:
                            bin_ = row['confidence_bin']
                            conf = safe_float_convert(row['mean_confidence'])
                            acc = safe_float_convert(row['accuracy'])
                            n_samples = safe_float_convert(row['num_samples'])
                            if not np.isnan(conf) and not np.isnan(acc) and not np.isnan(n_samples):
                                summary_data[bin_]['confidence'].append(conf)
                                summary_data[bin_]['accuracy'].append(acc)
                                summary_data[bin_]['num_samples'].append(n_samples)

                    # 提取ECE和AECE
                    last_row = df.iloc[-1]
                    if isinstance(last_row['mean_confidence'], str):
                        parts = last_row['mean_confidence'].split(',')
                        for part in parts:
                            if 'ECE=' in part:
                                ece = safe_float_convert(part.split('ECE=')[1])
                                if not np.isnan(ece):
                                    ece_data.append(ece)
                            if 'AECE=' in part:
                                aece = safe_float_convert(part.split('AECE=')[1].rstrip(','))
                                if not np.isnan(aece):
                                    aece_data.append(aece)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue

# 计算加权平均（按样本数加权）
avg_results = []
total_samples = 0
for bin_ in bins:
    if summary_data[bin_]['confidence']:
        # 计算每个数据点的权重（样本数）
        weights = np.array(summary_data[bin_]['num_samples'])
        total_bin_samples = np.sum(weights)
        total_samples += total_bin_samples

        # 加权平均
        avg_conf = np.average(summary_data[bin_]['confidence'], weights=weights)
        avg_acc = np.average(summary_data[bin_]['accuracy'], weights=weights)
        gap = avg_conf - avg_acc

        avg_results.append({
            'bin': bin_,
            'confidence': avg_conf,
            'accuracy': avg_acc,
            'gap': gap,
            'num_samples': total_bin_samples
        })

# 计算总体ECE和AECE（加权平均）
if ece_data and aece_data:
    avg_ece = np.mean(ece_data)
    avg_aece = np.mean(aece_data)
else:
    # 如果没有ECE数据，可以从bin数据计算
    total_ece = 0
    total_aece = 0
    for result in avg_results:
        total_ece += result['gap'] * result['num_samples']
        total_aece += abs(result['gap']) * result['num_samples']
    avg_ece = total_ece / total_samples if total_samples > 0 else 0
    avg_aece = total_aece / total_samples if total_samples > 0 else 0

# 打印结果
print("Aggregated Results Across All Datasets:")
for result in avg_results:
    print(
        f"{result['bin']}: Confidence={result['confidence']:.4f}, Accuracy={result['accuracy']:.4f}, Gap={result['gap']:.4f}, Samples={result['num_samples']}")
print(f"\nOverall ECE: {avg_ece:.4f}, Overall AECE: {avg_aece:.4f}")

# 准备绘图数据
if avg_results:
    # 提取 bin 的中点作为 x 坐标，并将其转为百分比
    bin_midpoints = [15, 25, 35, 45, 55, 65, 75, 85, 95]  # 百分比形式
    bin_width = 10  # 每个 bin 的宽度（百分比）
    confidences = [res['confidence'] * 100 for res in avg_results]  # 转换为百分比
    accuracies = [res['accuracy'] * 100 for res in avg_results]  # 转换为百分比

    # 创建可靠性图表
    plt.figure(figsize=(10, 8))
    plt.grid(True, linestyle='--', alpha=0.6)

    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 25,  # 默认字体大小
        'axes.titlesize': 25,  # 标题大小
        'axes.labelsize': 25,  # 坐标轴标签大小
        'xtick.labelsize': 25,  # x轴刻度数字大小
        'ytick.labelsize': 25,  # y轴刻度数字大小
        'legend.fontsize': 20  # 图例字体大小（稍小一点）
    })

    # 灰色对角线（完美校准线）
    plt.plot([0, 100], [0, 100], color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
             label='Perfect Calibration', zorder=1)

    for i, x in enumerate(bin_midpoints):
        left = x - bin_width / 2
        right = x + bin_width / 2
        acc = accuracies[i]
        conf = confidences[i]

        # 找出上下边界
        top = max(acc, conf)
        bottom = min(acc, conf)
        height = top - bottom

        # 绘制绿色 confidence 线
        plt.hlines(y=conf, xmin=left, xmax=right, colors='green', linewidth=6,
                   label='Confidence' if i == 0 else "", alpha=0.6, zorder=2)

        # 绘制蓝色 accuracy 线
        plt.hlines(y=acc, xmin=left, xmax=right, colors='blue', linewidth=6,
                   label='Accuracy' if i == 0 else "", alpha=0.6, zorder=2)

        # 绘制填充 Gap 矩形
        plt.bar(
            x=left,
            height=height,
            bottom=bottom,
            width=bin_width,
            align='edge',
            color='pink',
            alpha=0.6,
            edgecolor='none',
            zorder=3,
            label='Gap' if i == 0 else ""
        )

    # 添加总体ECE信息
    plt.text(
        95, 5,
        f'Overall ECE: {avg_ece:.4f}',
        ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8),
        fontsize=20  # ECE/AECE 文字大小
    )

    # 设置坐标轴和标签
    plt.xlabel('Confidence Bin (%)', fontsize=25)
    plt.ylabel('Accuracy / Confidence (%)', fontsize=25)
    plt.title('Calibration plots for Caltech101', fontsize=25)
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # 调整刻度标签大小
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    # 图例放左上角
    plt.legend(loc='upper left', fontsize=20)  # 图例字体稍小
    plt.tight_layout()
    plt.savefig('aggregated_reliability_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()