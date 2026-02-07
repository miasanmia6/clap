import pickle

with open('experiments/results/classification_EuroSAT_1.pkl', 'rb') as f:
    results = pickle.load(f)
res = results['results'][0]

print("Delta:", results['deltas'][0])
print("All available keys:")
for key in results['results'][0].keys():
    print("-", key)

print('完整协方差矩阵的结果：')
print("losses:", res['losses'])
print("train_nll_glm:", res['train_nll_glm'])
print("train_acc_glm:", res['train_acc_glm'])
print("train_ece_glm:", res['train_ece_glm'])
print("test_nll_glm:", res['test_nll_glm'])
print("valid_acc_glm:", res['valid_acc_glm'])
print("valid_ece_glm:", res['valid_ece_glm'])
print("test_acc_glm:", res['test_acc_glm'])
print("test_ece_glm:", res['test_ece_glm'])

print('对角协方差矩阵的结果：')
print("train_nll_glmd:", res['train_nll_glmd'])
print("train_acc_glmd:", res['train_acc_glmd'])
print("train_ece_glmd:", res['train_ece_glmd'])
print("test_nll_glmd:", res['test_nll_glmd'])
print("test_acc_glmd:", res['test_acc_glmd'])
print("test_ece_glmd:", res['test_ece_glmd'])
