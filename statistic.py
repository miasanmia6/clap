import json
from collections import defaultdict

# 你的 JSON 路径
json_path = "data/eurosat/split_zhou_EuroSAT.json"

# 读取 JSON 文件
with open(json_path, "r") as f:
    split = json.load(f)

test_set = split["test"]

# 统计每个类别的样本数
class_count = defaultdict(int)
for item in test_set:
    # item: [image_path, label_id, class_name]
    class_name = item[2]
    class_count[class_name] += 1

# 输出每个类的数量
for cls in sorted(class_count.keys()):
    print(f"{cls}: {class_count[cls]}")

# 输出总数
print(f"\nTotal test samples: {sum(class_count.values())}")
