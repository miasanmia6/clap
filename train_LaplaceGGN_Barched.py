# train_LaplaceGGN_batched.py
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="torch.cholesky is deprecated*")

import sys
import os
import torch.nn.functional as F
from torch import nn
import pickle
import numpy as np
import torch
import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import TensorDataset, DataLoader

from preds.optimizers import LaplaceGGN, get_diagonal_ggn
from preds.likelihoods import BernoulliLh, CategoricalLh
from preds.predictives import nn_sampling_predictive, linear_sampling_predictive
from preds.utils import acc, nll_cls, ece
from preds.refine import laplace_refine
from backpack import extend
from torch.optim.lr_scheduler import LambdaLR
import math


def analyze_batch_distribution(dataloader, dataset_name="Train"):
    """专业化的batch分布分析工具"""
    from collections import defaultdict
    import numpy as np

    print(f"\n===== {dataset_name} Set Batch Distribution Analysis =====")
    batch_stats = defaultdict(list)

    for batch_idx, (_, y_batch) in enumerate(dataloader):
        # 关键修改：添加.cpu()转换
        y_np = y_batch.cpu().numpy()  # 先转移到CPU再转numpy
        counts = dict(Counter(y_np))
        batch_stats['batch_idx'].append(batch_idx)
        batch_stats['class_dist'].append(counts)
        batch_stats['n_samples'].append(len(y_np))

        # 计算不平衡度指标
        class_probs = np.array(list(counts.values())) / len(y_np)
        imbalance = max(class_probs) / min(class_probs) if min(class_probs) > 0 else float('inf')
        batch_stats['imbalance_ratio'].append(imbalance)

    # 输出统计摘要
    print(f"Total batches: {len(batch_stats['batch_idx'])}")
    print(
        f"Average imbalance ratio: {np.mean(batch_stats['imbalance_ratio']):.2f} ± {np.std(batch_stats['imbalance_ratio']):.2f}")
    print(
        f"Max imbalance: {np.max(batch_stats['imbalance_ratio']):.2f} (Batch {np.argmax(batch_stats['imbalance_ratio'])})")

    # 可视化展示（可选）
    if 'matplotlib' in sys.modules:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.bar(batch_stats['batch_idx'], batch_stats['imbalance_ratio'])
        plt.xlabel('Batch Index')
        plt.ylabel('Imbalance Ratio')
        plt.title(f'{dataset_name} Set Batch Imbalance')
        plt.show()

def preds_glm(X, model, likelihood, mu, Sigma_chol, samples):
    gs = linear_sampling_predictive(X, model, likelihood, mu, Sigma_chol, mc_samples=samples)
    return gs.mean(dim=0)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class LinearCLIP(nn.Module):
    def __init__(self, base_text_features, learnable_logit_scale=False):
        super().__init__()
        K, D = base_text_features.shape
        self.output_size = K
        self.linear = nn.Linear(D, K, bias=False)
        normalized_weights = base_text_features / base_text_features.norm(dim=-1, keepdim=True)
        self.linear.weight.data = normalized_weights
        if learnable_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        else:
            self.register_buffer('logit_scale', torch.tensor(4.6052))

    def forward(self, image_features):
        image_features = F.normalize(image_features, dim=-1)
        logits = self.linear(image_features) * self.logit_scale.exp()
        return logits


def train(model, likelihood, train_loader, optimizer, n_epochs, warmup_epochs=100):
    losses = []
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, n_epochs)
    model.device = next(model.parameters()).device
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)

            def closure():
                optimizer.zero_grad()
                f = model(X_batch)
                return likelihood.log_likelihood(y_batch, f), f, y_batch

            loss = optimizer.step(closure)
            total_loss += loss
        scheduler.step()
        losses.append(total_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch}] loss={total_loss:.4f}, lr={current_lr:.6f}")
    optimizer.post_process(model, likelihood, train_loader)
    return losses


def batch_predict(model, likelihood, data_loader, optimizer=None, n_samples=1, verbose=True):
    """
    Batch预测函数（支持贝叶斯采样）并实时输出指标

    参数:
        model: 模型
        likelihood: 似然函数
        data_loader: 数据加载器
        optimizer: 可选，提供则从后验采样
        n_samples: 采样次数（仅贝叶斯模式有效）
        verbose: 是否打印每个batch的指标

    返回:
        all_logits: 所有batch的预测logits拼接结果
        all_labels: 所有batch的标签拼接结果
        batch_metrics: 每个batch的评估指标字典列表
    """
    model.eval()
    all_logits = []
    all_labels = []
    batch_metrics = []

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)

            if optimizer is not None and 'mu' in optimizer.state and 'Sigma_chol' in optimizer.state:
                # 贝叶斯预测模式
                mu = optimizer.state['mu']
                Sigma_chol = optimizer.state['Sigma_chol']

                batch_logits = []
                for _ in range(n_samples):
                    # 从后验分布采样参数
                    eps = torch.randn_like(mu)
                    theta_sample = mu + Sigma_chol @ eps

                    # 临时设置模型参数为采样值
                    original_params = parameters_to_vector(model.parameters())
                    vector_to_parameters(theta_sample, model.parameters())

                    # 计算预测
                    f = model(X_batch)
                    batch_logits.append(f)

                    # 恢复原始参数
                    vector_to_parameters(original_params, model.parameters())

                # 对多次采样取平均
                logits = torch.stack(batch_logits).mean(0)
            else:
                # 常规预测模式
                logits = model(X_batch)

            # 计算当前batch的指标
            probs = likelihood.inv_link(logits)
            batch_acc = acc(probs, y_batch, likelihood)
            batch_ece = ece(probs, y_batch, likelihood, bins=10)
            batch_nll = nll_cls(probs, y_batch, likelihood)

            # 确保指标是Python标量
            if torch.is_tensor(batch_acc):
                batch_acc = batch_acc.item()
            if torch.is_tensor(batch_ece):
                batch_ece = batch_ece.item()
            if torch.is_tensor(batch_nll):
                batch_nll = batch_nll.item()

            # 记录当前batch指标
            metrics = {
                'batch_idx': batch_idx,
                'acc': batch_acc,
                'ece': batch_ece,
                'nll': batch_nll
            }
            batch_metrics.append(metrics)

            if verbose:
                print(f"Batch {batch_idx + 1}/{len(data_loader)} - "
                      f"Acc: {batch_acc:.4f}, ECE: {batch_ece:.4f}, NLL: {batch_nll:.4f}")

            all_logits.append(logits)
            all_labels.append(y_batch)

    return torch.cat(all_logits), torch.cat(all_labels), batch_metrics


def evaluate_preds(p, y, likelihood, name, data, verbose=False):
    res = dict()
    res[f'{data}_nll_{name}'] = nll_cls(p, y, likelihood).item() if torch.is_tensor(
        nll_cls(p, y, likelihood)) else nll_cls(p, y, likelihood)
    res[f'{data}_acc_{name}'] = acc(p, y, likelihood).item() if torch.is_tensor(acc(p, y, likelihood)) else acc(p, y,
                                                                                                                likelihood)
    res[f'{data}_ece_{name}'] = ece(p, y, likelihood, bins=10).item() if torch.is_tensor(
        ece(p, y, likelihood, bins=10)) else ece(p, y, likelihood, bins=10)

    if verbose:
        print(f"\n{name} Results on {data}:")
        print(f"- NLL: {res[f'{data}_nll_{name}']:.4f}")
        print(f"- Accuracy: {res[f'{data}_acc_{name}']:.4f}")
        print(f"- ECE: {res[f'{data}_ece_{name}']:.4f}")

    return res


def inference(ds_train, ds_test, ds_valid, base_text_features, prior_prec, lr, n_epochs, device, seed, n_samples=1000):
    train_loader = DataLoader(ds_train, batch_size=256, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=512)
    valid_loader = DataLoader(ds_valid, batch_size=512)

    analyze_batch_distribution(train_loader, "Train")
    analyze_batch_distribution(test_loader, "Test")

    res = dict()
    torch.manual_seed(seed)
    K = base_text_features.shape[0]
    likelihood = BernoulliLh() if K == 2 else CategoricalLh()
    model = LinearCLIP(base_text_features.to(device))
    model = extend(model).to(device)
    model.device = device
    prior_mu = base_text_features.flatten().to(device)
    optimizer = LaplaceGGN(model, lr=lr, prior_prec=prior_prec, prior_mu=prior_mu)
    res['losses'] = train(model, likelihood, train_loader, optimizer, n_epochs)

    theta_star = parameters_to_vector(model.parameters()).detach()
    Sigmad, Sigma_chold = get_diagonal_ggn(optimizer)
    Sigma_chol = optimizer.state['Sigma_chol']

    # MAP预测（使用batch_predict）
    print("\nEvaluating MAP predictions:")
    fs_train, y_train, _ = batch_predict(model, likelihood, train_loader, verbose=True)
    fs_test, y_test, _ = batch_predict(model, likelihood, test_loader, verbose=True)
    fs_valid, y_valid, _ = batch_predict(model, likelihood, valid_loader, verbose=True)

    res.update(evaluate_preds(likelihood.inv_link(fs_train), y_train, likelihood, 'map', 'train', verbose=True))
    res.update(evaluate_preds(likelihood.inv_link(fs_test), y_test, likelihood, 'map', 'test', verbose=True))
    res.update(evaluate_preds(likelihood.inv_link(fs_valid), y_valid, likelihood, 'map', 'valid', verbose=True))

    # Laplace refine
    m, S_chol, S_chold, losses = laplace_refine(model, ds_train.tensors[0].to(device), ds_train.tensors[1].to(device),
                                                likelihood, prior_prec)
    res['losses_lap'] = losses

    # Sampling methods
    methods = [
        ('glm', theta_star, Sigma_chol),
        ('glmd', theta_star, Sigma_chold),
        ('glmLap', m, S_chol),
        ('glmLapd', m, S_chold),
    ]

    for name, mu, chol in methods:
        print(f"\nEvaluating {name} predictions:")
        # 使用batch_predict进行预测
        fs_train, y_train, batch_metrics = batch_predict(model, likelihood, train_loader, optimizer=None,
                                                         n_samples=n_samples, verbose=True)
        fs_test, y_test, _ = batch_predict(model, likelihood, test_loader, optimizer=None, n_samples=n_samples,
                                           verbose=True)
        fs_valid, y_valid, _ = batch_predict(model, likelihood, valid_loader, optimizer=None, n_samples=n_samples,
                                             verbose=True)

        # 保存batch指标
        res[f'{name}_batch_metrics'] = batch_metrics

        # 使用preds_glm计算预测概率
        p_train = preds_glm(ds_train.tensors[0].to(device), model, likelihood, mu, chol, n_samples)
        p_test = preds_glm(ds_test.tensors[0].to(device), model, likelihood, mu, chol, n_samples)
        p_valid = preds_glm(ds_valid.tensors[0].to(device), model, likelihood, mu, chol, n_samples)

        res.update(evaluate_preds(p_train, ds_train.tensors[1].to(device), likelihood, name, 'train', verbose=True))
        res.update(evaluate_preds(p_test, ds_test.tensors[1].to(device), likelihood, name, 'test', verbose=True))
        res.update(evaluate_preds(p_valid, ds_valid.tensors[1].to(device), likelihood, name, 'valid', verbose=True))

    return res

def load_data_and_features(dataset_name, device):
    feature_path = f"features/features_{dataset_name.lower()}.pt"
    data = torch.load(feature_path)
    text_path = f"features/base_text_features_{dataset_name.lower()}.pt"
    text_data = torch.load(text_path)
    features_train = data["features_train"].float()
    labels_train = data["labels_train"].long()
    base_text_features = text_data["base_text_features"].float()
    ds_train = TensorDataset(features_train.to(device), labels_train.to(device))
    ds_test = TensorDataset(data["features_test"].float().to(device), data["labels_test"].long().to(device))
    ds_valid = TensorDataset(data["features_test"].float().to(device), data["labels_test"].long().to(device))
    return ds_train, ds_test, ds_valid, base_text_features.to(device)


def main(dataset="EuroSAT", seed=42, n_epochs=1000, lr=1e-3, prior_prec=100.0, n_samples=1000, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_train, ds_test, ds_valid, base_text_features = load_data_and_features(dataset, device)
    results = inference(ds_train, ds_test, ds_valid, base_text_features, prior_prec, lr, n_epochs, device, seed,
                        n_samples)
    resdict = {
        'results': results,
        'prior_prec': prior_prec,
        'N_train': len(ds_train),
        'N_test': len(ds_test),
        'K': base_text_features.shape[0]
    }
    os.makedirs('experiments/results_batched', exist_ok=True)
    with open(f'experiments/results_batched/classification_{dataset}_{seed}.pkl', 'wb') as f:
        pickle.dump(resdict, f)


if __name__ == '__main__':
    import argparse

    print("\n===== Initializing Laplace-GGN Training =====")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EuroSAT')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--prior_prec', type=float, default=100.0)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"- {k}: {v}")
    try:
        main(**vars(args))
        print("\n===== Training Completed Successfully =====")
    except Exception as e:
        print(f"\n!!! Training Failed: {str(e)} !!!")
        raise