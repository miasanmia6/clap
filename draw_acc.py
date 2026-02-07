import matplotlib.pyplot as plt

# 模型名称，年份作为下角标
models = [
    "LP$_{ICML21}$", "TipA$_{ECCV22}$", "TipA-f-$_{ECCV22}$", "CrossModal$_{CVPR23}$",
    "TaskRes$_{CVPR23}$", "LP++$_{CVPR24}$", "CLAP$_{CVPR24}$", "BA$_{CVPR25}$", "Ours"
]

# x轴：训练样本比例或阶段
x = [1, 2, 4, 8, 16, 32]

# 每个模型对应的ACC数据
acc_data = [
    [32.76, 42.33, 53.63, 61.81, 69.02, 73.69],
    [59.19, 60.81, 62.50, 65.04, 66.98, 64.38],
    [61.03, 63.66, 66.37, 70.34, 73.29, 74.73],
    [63.08, 65.98, 69.08, 72.78, 75.69, 78.06],
    [63.60, 65.93, 68.48, 72.23, 75.25, 77.61],
    [62.88, 66.04, 69.63, 72.38, 75.25, 76.91],
    [63.68, 66.70, 69.85, 73.10, 75.63, 77.01],
    [62.23, 65.30, 68.83, 72.43, 75.62, 78.41],
    [62.66, 67.05, 70.13, 72.93, 75.44, 77.10]
]

# 设置颜色和标记风格
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#d62728']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']

# 设置正方形画布
plt.figure(figsize=(11,11))

for i, model in enumerate(models):
    plt.plot(x, acc_data[i], label=model, color=colors[i], marker=markers[i], linewidth=5, markersize=15)

# 设置轴标签和标题
plt.xlabel("Number of Shots", fontsize=25)
plt.ylabel("Accuracy (%)", fontsize=25)
plt.title("Model Comparison on ACC", fontsize=25)

# 设置刻度字体大小
plt.xticks(x, fontsize=25)
plt.yticks(fontsize=25)

plt.grid(True, linestyle='--', alpha=2)
plt.legend(fontsize=25, ncol=2)  # 图例分两列

# 修改图像边框粗细和颜色
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig('C:\\Users\\123\\Pictures\\CVPR\\ACC.jpg')
plt.show()

