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
    [47.25, 37.19, 32.28, 16.86, 10.01, 6.18],
    [3.83, 4.02, 4.46, 6.13, 9.13, 17.59],
    [5.24, 5.50, 4.65, 4.58, 5.62, 8.93],
    [5.24, 5.24, 4.54, 4.68, 4.24, 3.71],
    [5.28, 5.90, 6.43, 7.03, 6.65, 5.91],
    [18.14, 17.53, 17.21, 12.23, 9.21, 6.66],
    [5.22, 5.38, 6.36, 7.50, 8.59, 9.19],
    [6.15, 4.89, 3.64, 3.26, 3.03, 2.91],
    [2.40, 1.79, 1.50, 1.40, 1.51, 1.45]
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
plt.ylabel("ECE (%)", fontsize=25)
plt.title("Model Comparison on ECE", fontsize=25)

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
plt.savefig('C:\\Users\\123\\Pictures\\CVPR\\ECE.jpg')
plt.show()

