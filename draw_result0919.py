import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 示例数据（每 shot 平均指标）
# -----------------------------
shots = [1, 2, 4, 8, 16, 32]

# 模型名称
models = [
    "LP$_{ICML21}$", "TipA$_{ECCV22}$", "TipA-f-$_{ECCV22}$",
    "CrossModal$_{CVPR23}$", "TaskRes$_{CVPR23}$", "LP++$_{CVPR24}$",
    "CLAP$_{CVPR24}$", "BA$_{CVPR25}$", "Ours"
]

# 每个模型的ACC（%）
acc_data = [
    [32.76, 42.33, 53.63, 61.81, 69.02, 73.90],
    [59.19, 60.81, 62.50, 65.04, 66.98, 64.48],
    [61.03, 63.66, 66.37, 70.34, 73.49, 74.73],
    [63.08, 65.98, 69.08, 71.88, 75.79, 78.17],
    [63.60, 65.93, 68.48, 71.89, 75.25, 77.61],
    [62.88, 66.04, 69.63, 72.38, 75.25, 77.01],
    [63.68, 66.70, 69.85, 73.01, 75.63, 76.91],
    [62.23, 65.30, 68.83, 72.43, 75.62, 77.61],
    [62.66, 67.05, 70.13, 72.93, 76.04, 77.00]
]

# 每个模型的ECE（%）
ece_data = [
    [13.47, 12.33, 10.50, 9.20, 7.90, 6.50],
    [1.35, 1.40, 1.45, 1.50, 1.55, 1.60],
    [1.32, 1.30, 1.28, 1.25, 1.23, 1.20],
    [3.11, 3.05, 2.98, 2.90, 2.85, 2.80],
    [4.47, 4.40, 4.30, 4.20, 4.10, 4.00],
    [16.06, 15.50, 14.80, 14.20, 13.70, 13.20],
    [3.55, 3.50, 3.45, 3.40, 3.35, 3.30],
    [1.30, 1.28, 1.25, 1.22, 1.20, 1.18],
    [0.86, 0.85, 0.83, 0.80, 0.78, 0.75]
]

# 每个模型的AECE（%）
aece_data = [
    [12.0, 11.2, 10.0, 9.1, 8.2, 7.3],
    [1.2, 1.3, 1.3, 1.4, 1.4, 1.5],
    [1.1, 1.1, 1.1, 1.0, 1.0, 0.9],
    [2.5, 2.4, 2.3, 2.2, 2.1, 2.0],
    [3.8, 3.7, 3.6, 3.5, 3.4, 3.3],
    [14.0, 13.5, 13.0, 12.5, 12.0, 11.5],
    [2.9, 2.8, 2.7, 2.6, 2.5, 2.4],
    [1.1, 1.0, 1.0, 0.9, 0.9, 0.8],
    [0.8, 0.7, 0.7, 0.6, 0.6, 0.5]
]

# -----------------------------
# 设置颜色和标记
# -----------------------------
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#d62728']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']

# -----------------------------
# 绘制折线图（ACC, ECE, AECE）
# -----------------------------
fig, ax1 = plt.subplots(figsize=(10,10))  # 正方形图

for i, model in enumerate(models):
    ax1.plot(shots, acc_data[i], label=f"{model} ACC", color=colors[i],
             marker=markers[i], linewidth=2, markersize=7)
    ax1.plot(shots, ece_data[i], label=f"{model} ECE", color=colors[i],
             marker=markers[i], linewidth=1.5, markersize=5, linestyle='--')
    ax1.plot(shots, aece_data[i], label=f"{model} AECE", color=colors[i],
             marker=markers[i], linewidth=1.5, markersize=5, linestyle=':')

ax1.set_xlabel("Number of Shots", fontsize=25)
ax1.set_ylabel("ACC / ECE / AECE (%)", fontsize=25)
ax1.set_title("Model Prediction Accuracy and Calibration", fontsize=25)
ax1.set_xticks(shots)
ax1.tick_params(axis='both', labelsize=20)
ax1.grid(True, linestyle='--', alpha=0.5)

# 图例
ax1.legend(fontsize=12, ncol=2)
plt.tight_layout()
plt.show()

# -----------------------------
# 总体校准指标条形图
# -----------------------------
# 取最后一个 shot 的 ACC / ECE / AECE 作为代表
final_acc = [acc[-1] for acc in acc_data]
final_ece = [ece[-1] for ece in ece_data]
final_aece = [aece[-1] for aece in aece_data]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10,10))  # 正方形
rects1 = ax.bar(x - width, final_acc, width, label='ACC', color='#1f77b4')
rects2 = ax.bar(x, final_ece, width, label='ECE', color='#ff7f0e')
rects3 = ax.bar(x + width, final_aece, width, label='AECE', color='#2ca02c')

ax.set_ylabel('Percentage (%)', fontsize=25)
ax.set_title('Overall Accuracy and Calibration', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=18, rotation=30, ha='right')
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=15)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
