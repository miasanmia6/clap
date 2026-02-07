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
    [47.24, 37.19, 25.79, 16.85, 9.98, 6.15],
    [3.74, 4.03, 4.45, 6.12, 9.05, 17.57],
    [5.19, 5.45, 4.60, 4.47, 5.56, 8.82],
    [5.30, 5.24, 4.55, 4.56, 4.21, 3.63],
    [5.28, 5.86, 6.48, 7.04, 6.64, 5.83],
    [18.14, 17.56, 17.19, 12.25, 9.25, 6.66],
    [5.19, 5.33, 6.34, 7.44, 8.59, 9.22],
    [6.13, 4.83, 3.57, 3.20, 2.97, 2.83],
    [2.22, 1.79, 1.50, 1.40, 1.51, 1.45]
]

# 设置颜色和标记风格
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#d62728']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']

plt.figure(figsize=(11,11))

for i, model in enumerate(models):
    plt.plot(x, acc_data[i], label=model, color=colors[i], marker=markers[i], linewidth=5, markersize=15)

# 设置轴标签和标题
plt.xlabel("Number of Shots", fontsize=25)
plt.ylabel("AECE (%)", fontsize=25)
plt.title("Model Comparison on AECE", fontsize=25)

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
plt.savefig('C:\\Users\\123\\Pictures\\CVPR\\AECE.jpg')
plt.show()

