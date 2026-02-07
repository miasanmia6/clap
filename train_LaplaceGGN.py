import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.cholesky is deprecated*")

import sys
import os
import torch.nn.functional as F
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np
import torch
import tqdm
from torch.nn.utils import parameters_to_vector
from torch.utils.data import TensorDataset

from preds.optimizers import LaplaceGGN, get_diagonal_ggn
from preds.likelihoods import BernoulliLh, CategoricalLh
from preds.predictives import nn_sampling_predictive, linear_sampling_predictive
from preds.utils import acc, nll_cls, ece
from preds.mfvi import run_bbb
from preds.refine import laplace_refine, vi_refine, vi_diag_refine
from backpack import extend
from backpack import backpack
from backpack.extensions import BatchGrad

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        # Cosine decay
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class LinearCLIP(nn.Module):
    def __init__(self, base_text_features, learnable_logit_scale=False):
        super().__init__()
        K, D = base_text_features.shape  # K: 类别数, D: 特征维度
        self.output_size = base_text_features.shape[0]

        # 将 normalized prototype 向量加载为 Linear 层的权重
        self.linear = nn.Linear(D, K, bias=False)
        normalized_weights = base_text_features / base_text_features.norm(dim=-1, keepdim=True)
        self.linear.weight.data = normalized_weights

        if learnable_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(4.6052))  # ln(100)
        else:
            self.register_buffer('logit_scale', torch.tensor(4.6052))

    def forward(self, image_features):
        image_features = F.normalize(image_features, dim=-1)  # L2 norm
        logits = self.linear(image_features) * self.logit_scale.exp()
        return logits

def train(model, likelihood, X_train, y_train, optimizer, n_epochs, warmup_epochs=100):
    losses = []
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, n_epochs)

    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()
            f = model(X_train)
            return likelihood.log_likelihood(y_train, f),f,y_train

        loss = optimizer.step(closure)
        scheduler.step()
        losses.append(loss)

        if epoch % 1 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch}] loss={loss:.4f}, lr={current_lr:.6f}")

    optimizer.post_process(model, likelihood, [(X_train, y_train)])
    return losses


def preds_glm(X, model, likelihood, mu, Sigma_chol, samples):
    # print("Checking mu/Sigma_chol inside preds_glm...")
    # print("mu has nan?", torch.isnan(mu).any().item())
    # print("Sigma_chol has nan?", torch.isnan(Sigma_chol).any().item())

    gs = linear_sampling_predictive(X, model, likelihood, mu, Sigma_chol, mc_samples=samples)

    # print("output has nan?", torch.isnan(gs).any().item())
    return gs.mean(dim=0)


def preds_nn(X, model, likelihood, mu, Sigma_chol, samples):
    gs = nn_sampling_predictive(X, model, likelihood, mu, Sigma_chol, mc_samples=samples)
    return gs.mean(dim=0)


def evaluate(p, y, likelihood, name, data):
    if torch.isnan(p).any():
        print(f"[Warning] Predictions contain NaN for {data}_{name}")
    res = dict()
    res[f'{data}_nll_{name}'] = nll_cls(p, y, likelihood)
    res[f'{data}_acc_{name}'] = acc(p, y, likelihood)
    print(str(name)+'_acc_test_'+str(data)+':',res[f'{data}_acc_{name}'])
    res[f'{data}_ece_{name}'] = ece(p, y, likelihood, bins=10)
    return res


def inference(ds_train, ds_test, ds_valid, base_text_features, prior_prec, lr, n_epochs, device, seed, n_samples=1000):
    X_train, y_train = ds_train.tensors[0].to(device), ds_train.tensors[1].to(device)
    X_test, y_test = ds_test.tensors[0].to(device), ds_test.tensors[1].to(device)
    X_valid, y_valid = ds_valid.tensors[0].to(device), ds_valid.tensors[1].to(device)

    res = dict()
    torch.manual_seed(seed)

    K = base_text_features.shape[0]
    likelihood = BernoulliLh() if K == 2 else CategoricalLh()

    model = LinearCLIP(base_text_features.to(device))
    model = extend(model)  # 注意只能 extend 一次
    model = model.to(device)
    prior_mu = base_text_features.flatten().to(device)
    optimizer = LaplaceGGN(model, lr=lr, prior_prec=prior_prec, prior_mu=prior_mu)

    res['losses'] = train(model, likelihood, X_train, y_train, optimizer, n_epochs)


    theta_star = parameters_to_vector(model.parameters()).detach()
    Sigmad, Sigma_chold = get_diagonal_ggn(optimizer)
    Sigma_chol = optimizer.state['Sigma_chol']

    # MAP evaluation
    fs_train = likelihood.inv_link(model(X_train).detach())
    fs_test = likelihood.inv_link(model(X_test).detach())
    fs_valid = likelihood.inv_link(model(X_valid).detach())
    res.update(evaluate(fs_train, y_train, likelihood, 'map', 'train'))
    res.update(evaluate(fs_test, y_test, likelihood, 'map', 'test'))
    res.update(evaluate(fs_valid, y_valid, likelihood, 'map', 'valid'))

    # Full Laplace
    m, S_chol, S_chold, losses = laplace_refine(model, X_train, y_train, likelihood, prior_prec)
    res['losses_lap'] = losses


    # Collect all predictions
    methods = [
        ('glm', theta_star, Sigma_chol),
        ('glmd', theta_star, Sigma_chold),
        ('glmLap', m, S_chol),
        ('glmLapd', m, S_chold),
    ]

    for name, mu, chol in methods:
        fs_train = preds_glm(X_train, model, likelihood, mu, chol, n_samples)
        fs_test = preds_glm(X_test, model, likelihood, mu, chol, n_samples)
        fs_valid = preds_glm(X_valid, model, likelihood, mu, chol, n_samples)
        # print(f"[Debug] NaN check for prediction")
        # print(f"fs_train has nan? {torch.isnan(fs_train).any().item()}")
        res.update(evaluate(fs_train, y_train, likelihood, name, 'train'))
        res.update(evaluate(fs_test, y_test, likelihood, name, 'test'))
        res.update(evaluate(fs_valid, y_valid, likelihood, name, 'valid'))

    return res

def load_data_and_features(dataset_name, device):
    # Load image features
    feature_path = f"features/features_{dataset_name.lower()}.pt"
    data = torch.load(feature_path)

    # Load text features
    text_path = f"features/base_text_features_{dataset_name.lower()}.pt"
    text_data = torch.load(text_path)

    # 强制转换为float32
    features_train = data["features_train"].float()
    labels_train = data["labels_train"].long()  # 标签转为long
    base_text_features = text_data["base_text_features"].float()

    # Create datasets
    ds_train = TensorDataset(features_train.to(device), labels_train.to(device))
    ds_test = TensorDataset(data["features_test"].float().to(device),
                            data["labels_test"].long().to(device))
    ds_valid = TensorDataset(data["features_test"].float().to(device),
                             data["labels_test"].long().to(device))

    return ds_train, ds_test, ds_valid, base_text_features.to(device)


def main(dataset="EuroSAT", seed=42, n_epochs=1000, lr=1e-3, n_deltas=10,
         logd_min=-2.0, logd_max=2.0, n_samples=1000, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    ds_train, ds_test, ds_valid, base_text_features = load_data_and_features(dataset, device)

    # Run experiments
    deltas = [100.0]
    results = []
    for delta in tqdm.tqdm(deltas):
        res = inference(ds_train, ds_test, ds_valid, base_text_features, delta,
                        lr, n_epochs, device, seed, n_samples)
        results.append(res)

    # Save results
    resdict = {
        'results': results,
        'deltas': deltas,
        'N_train': len(ds_train),
        'N_test': len(ds_test),
        'K': base_text_features.shape[0]
    }

    os.makedirs('experiments/results', exist_ok=True)
    with open(f'experiments/results/classification_{dataset}_{seed}.pkl', 'wb') as f:
        pickle.dump(resdict, f)


if __name__ == '__main__':
    import argparse

    print("\n===== Initializing Laplace-GGN Training =====")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EuroSAT')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.1) # 1e-3
    parser.add_argument('--n_deltas', type=int, default=10)
    parser.add_argument('--logd_min', type=float, default=-2.0)
    parser.add_argument('--logd_max', type=float, default=2.0)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    print("Configuration:")
    print(f"- Dataset: {args.dataset}")
    print(f"- Seed: {args.seed}")
    print(f"- Epochs: {args.n_epochs}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    try:
        main(**vars(args))
        print("\n===== Training Completed Successfully =====")
    except Exception as e:
        print(f"\n!!! Training Failed: {str(e)} !!!")
        raise