import torch

# 加载保存的特征
data = torch.load("./features/saved_features_eurosat.pt")
features_train = data["features_train"]  # [N, D]
labels_train = data["labels_train"]      # [N]
features_test = data["features_test"]    # [M, D]
labels_test = data["labels_test"]        # [M]

print(
    f"Train: {features_train.shape}, {labels_train.shape}\n"
    f"Test: {features_test.shape}, {labels_test.shape}"
)