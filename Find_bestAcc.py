import re
from collections import defaultdict


def parse_experiment_results(file_path):
    # 存储最终结果 {(seed, num_shots): max_acc_test}
    results = defaultdict(float)
    # 当前正在处理的实验的参数
    current_seed = None
    current_num_shots = None
    current_max_acc = 0.0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 检查是否开始新的实验部分
            if line.startswith('** Arguments **'):
                # 保存上一个实验的结果
                if current_seed is not None and current_num_shots is not None:
                    key = (current_seed, current_num_shots)
                    if current_max_acc > results[key]:
                        results[key] = current_max_acc
                # 重置当前实验变量
                current_max_acc = 0.0
                current_seed = None
                current_num_shots = None
            # 提取seed参数
            elif line.startswith('seed:'):
                current_seed = int(line.split(':')[1].strip())
            # 提取num_shots参数
            elif line.startswith('num_shots:'):
                current_num_shots = int(line.split(':')[1].strip())
            # 提取acc_test值
            elif 'acc_test' in line:
                # 使用正则表达式提取acc_test值
                match = re.search(r'acc_test\s+([\d.]+)\s+\(([\d.]+)\)', line)
                if match:
                    acc_test = float(match.group(2))
                    if acc_test > current_max_acc:
                        current_max_acc = acc_test

    # 处理最后一个实验
    if current_seed is not None and current_num_shots is not None:
        key = (current_seed, current_num_shots)
        if current_max_acc > results[key]:
            results[key] = current_max_acc

    return results


def print_results(results):
    # 按seed和num_shots排序输出结果
    sorted_keys = sorted(results.keys(), key=lambda x: (x[0], x[1]))
    print("Seed\tShots\tBest Test Accuracy")
    print("-----------------------------")
    for key in sorted_keys:
        seed, num_shots = key
        print(f"{seed}\t{num_shots}\t{results[key]:.4f}")


# 使用示例
file_path = '/root/autodl-tmp/CLAP-LaplaceGGN/LaplaceGGN/FINAL/debug/oxford_pets_l2/SGD_lr1e-1_B256_ep300/kl/1shots/1seed/log.txt-2025-06-13-16-59-38'  # 替换为你的文件路径
results = parse_experiment_results(file_path)
print_results(results)
