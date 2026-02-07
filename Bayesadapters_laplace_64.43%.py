import os
import os.path as osp
from re import template
from torch.autograd.functional import jacobian
import torch
import gc
import math
import time
import os.path as osp
import datetime
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import gc
import weakref
from torch.autograd.functional import jacobian

from dassl.engine import TRAINER_REGISTRY, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, AverageMeter, MetricMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datasets.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.backends.cuda.matmul.allow_tf32 = False  # 防止隐式图构建

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def analyze_confidence_distribution(logits, num_bins=20, save_path="confidence_hist.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    probs = torch.softmax(logits, dim=1)
    confidences = probs.max(dim=1).values.detach().cpu().numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(confidences, bins=num_bins, color='skyblue', edgecolor='black')
    plt.title("Confidence Distribution (max softmax per sample)")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)  # ✅ 保存图像
    print(f"[Saved confidence histogram to {save_path}]")

    print(f"Mean Confidence: {confidences.mean():.4f}")
    print(f"Min Confidence:  {confidences.min():.4f}")
    print(f"Max Confidence:  {confidences.max():.4f}")
    print(f"Percent > 0.9:   {(confidences > 0.9).mean() * 100:.2f}%")
    print(f"Percent < 0.5:   {(confidences < 0.5).mean() * 100:.2f}%")


def find_optimal_temperature(logits, labels):
    import torch.nn.functional as F
    from torch import optim

    logits = logits.detach()  # ✅ 防止重复反向传播错误
    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    labels = labels.to(logits.device)

    def loss_fn():
        scaled_logits = logits / temperature
        return F.cross_entropy(scaled_logits, labels)

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.detach()


def print_gpu_memory(note=""):
    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"[GPU MEM] {note} - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def _get_base_text_features(cfg, classnames, clip_model, text_encoder, pretrained_projection=None):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()

        if pretrained_projection is not None:
            # Load pretrained projection from TaskResidual Work
            pretrained_text_projection = torch.load(pretrained_projection)

            # Move weight to current CLIP model
            state_dict = text_encoder.state_dict()
            state_dict['text_projection'] = pretrained_text_projection['state_dict']['weight'].t()
            text_encoder.load_state_dict(state_dict)
            print(">> Pretrained text encoder loaded!")
            params = pretrained_text_projection['state_dict']['weight'].size(0) * \
                     pretrained_text_projection['state_dict']['weight'].size(1)
            print(">> Text projection parameters: ", params)
            print(pretrained_text_projection['state_dict'].keys())

    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            prototype = text_encoder(embeddings.cuda(), tokens.cuda())
            text_embeddings.append(prototype)

    text_embeddings = torch.stack(text_embeddings)
    text_embeddings_avg = text_embeddings.mean(1)
    return text_embeddings_avg.to(device), text_embeddings


class AdapterMethod(nn.Module):
    def __init__(self, cfg, clip_model, base_text_features):
        super().__init__()
        self.device = clip_model.dtype
        self.logit_scale = clip_model.logit_scale
        self.initialization = cfg.TRAINER.LaplaceBayesADAPTER.INIT
        print("Initialization method:", cfg.TRAINER.LaplaceBayesADAPTER.INIT)
        self.apply_constraint = cfg.TRAINER.LaplaceBayesADAPTER.CONSTRAINT
        self.distance = cfg.TRAINER.LaplaceBayesADAPTER.CONSTRAINT  # "l2"
        self.register_buffer("base_text_features", base_text_features)
        self.alpha_constraint = torch.zeros((base_text_features.shape[0])).to(self.device)
        self.base_text_features = base_text_features
        self.augmentations = True  # True
        self.epochs_aumentation = 20  # 20
        # 新增 Laplace-GGN 相关参数
        self.register_buffer("ggn_cov", None)  # 协方差矩阵 [K, D, K, D]
        self.register_buffer("prior_mu", self.base_text_features.clone().to('cuda:0').flatten())
        self.prototypes_map = None
        self.posterior_chol = None
        self.precision = None
        self.n_samples = 1000  # MC 采样次数

        if self.initialization == "RANDOM":  # Randomly initialized Linear Probing
            print("Using RANDOM initialization in Linear Probing", end="\n")
            self.prototypes = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(base_text_features.shape)))
        elif "ZS" in self.initialization:  # Linear Probe initialized with zero-shot weights
            print("Using Zero-Shot initialization in Linear Probing", end="\n")
            self.prototypes = nn.Parameter(base_text_features.clone())
            K, D = self.prototypes.shape
            self.register_buffer("prior_var", torch.full((K * D,), 0.05))#0.5
        else:
            print("Initialization for Linear Probing not implemented")
            assert False

        if self.apply_constraint != "none":
            print("Applying constraint to the logistic regression weights: " + str(self.distance))

    def sample_prototypes(self, n_samples=1):
        """从 Laplace-GGN 后验中采样原型"""
        if self.ggn_cov is None:
            return self.prototypes.unsqueeze(0)  # 未计算协方差时返回 MAP 估计

        # 获取原型和协方差矩阵的形状
        K, D = self.prototypes_map.shape  # [num_classes, feature_dim]

        # 从 N(prototypes_map, ggn_cov) 采样
        prototypes_samples = []
        for _ in range(n_samples):
            # 生成与原型形状相同的随机噪声
            noise = torch.randn_like(self.prototypes_map)  # [K, D]

            # 获取协方差矩阵的对角元素
            # 假设协方差是对角矩阵，形状应为 [K, D]
            cov_diag = torch.diagonal(self.ggn_cov.reshape(K * D, K * D)).reshape(K, D)

            # 确保 cov_diag 的形状与 prototypes_map 相同
            if cov_diag.shape != self.prototypes_map.shape:
                # 如果协方差矩阵是 [K, D, K, D]，取每个类的对角协方差
                cov_diag = torch.diagonal(self.ggn_cov, dim1=0, dim2=2).permute(2, 0, 1)  # [K, D, D]
                cov_diag = torch.diagonal(cov_diag, dim1=1, dim2=2)  # [K, D]

            # 确保 cov_diag 非负
            cov_diag = torch.clamp(cov_diag, min=0)

            # 采样
            sample = self.prototypes_map + torch.sqrt(cov_diag) * noise
            prototypes_samples.append(sample)

        return torch.stack(prototypes_samples)  # [n_samples, K, D]

    def init_MultiModal(self):
        print("Using Zero-Shot initialization in Linear Probing", end="\n")
        self.prototypes = nn.Parameter(self.base_text_features.clone())

    def zero_shot_constraint(self):

        # Compute constraint
        if "l2" in self.apply_constraint:
            disimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(-1)
        elif "cosine" in self.apply_constraint:
            disimilitude = (1 - torch.nn.functional.cosine_similarity(self.prototypes, self.base_text_features.clone()))
        else:
            print("Dissimilitude metric for constraint not implemented")
            assert False

        return torch.mean(self.alpha_constraint * disimilitude)

    def init_lagrangian_multipliers(self, labels_ds, logits_ds):
        if "balanced" in self.apply_constraint:
            performance = torch.ones(logits_ds.shape[-1]).to(torch.float)
        else:
            with torch.no_grad():

                # Get one-hot encoding ground-truth
                labels_one_hot = torch.nn.functional.one_hot(labels_ds).cpu()

                # Get zero_shot performance
                performance = torch.diag(torch.softmax(logits_ds, -1).t() @ labels_one_hot.to(torch.float32)) / \
                              labels_one_hot.sum(0)

                if "corrected" in self.apply_constraint:
                    performance *= (logits_ds.shape[-1] / torch.sum(performance).item())
                if "constant" in self.apply_constraint:
                    performance = torch.ones(logits_ds.shape[-1]).to(torch.float) * torch.mean(performance).item()

        # set new alphas
        self.alpha_constraint = torch.clone(performance).to(self.device)
        self.penalty_parameter = torch.zeros_like(self.alpha_constraint).to(self.device)

    def outer_step(self):
        def phr(h, lambd, rho):
            x = lambd + rho * h
            y_sup = 1 / (2 * rho) * (x ** 2 - lambd ** 2)
            y_inf = - 1 / (2 * rho) * (lambd ** 2)

            grad_y_sup = x
            grad_y_inf = torch.zeros_like(h)

            sup = x >= 0
            return (
                torch.where(sup, y_sup, y_inf),
                torch.where(sup, grad_y_sup, grad_y_inf)
            )

        print("Outer step on Augmented Lagrangian Multiplier")

        # Cmpute current constraints
        disimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(-1)

        # Compute phr
        phr_value, phr_grad = phr(disimilitude, self.alpha_constraint, self.penalty_parameter)

        # Update lagrangian multipliers
        self.alpha_constraint = phr_grad.detach().clone()

        # Update penalty parameters rho
        self.penalty_parameter = disimilitude.detach().clone()

        print("New lagrangian multipliers:")
        print(self.alpha_constraint[0:5].detach().cpu().numpy())

    def forward(self):
        return self.prototypes


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype  # float16
        text_encoder = TextEncoder(clip_model)

        # For TaskRes (Yu et al.) enhanced base - or regular CLIP base
        if cfg.TRAINER.LaplaceBayesADAPTER.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features, text_embeddings_all = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        else:
            print(">> Use enhanced base!")
            base_text_features, text_embeddings_all = _get_base_text_features(
                cfg, classnames, clip_model, text_encoder, cfg.TRAINER.TaskRes.ENHANCED_BASE)

        self.text_embeddings_all = text_embeddings_all
        self.adapter = AdapterMethod(cfg, clip_model, base_text_features)

    def compute_ggn_diag_covariance(self, feature_train, label_train, batch_size=16, save_path=None):
        import gc
        from torch.autograd.functional import jacobian
        from tqdm import tqdm
        import torch.nn.functional as F
        import torch

        # 计算后验前检查原型
        print("计算后验前检查原型--Prototypes before posterior calculation:")
        print(self.adapter.prototypes.data[:2, :5])

        # 开启 prototype 参数梯度
        self.adapter.prototypes.requires_grad_(True)
        K, D = self.adapter.prototypes.shape
        device = feature_train.device

        # 直接移动到 device
        feature_train = feature_train.to(device, non_blocking=True)
        label_train = label_train.to(device, non_blocking=True)

        # 初始化 GGN 对角项 和 一阶梯度项
        GGN_diag = torch.zeros(K * D, device='cpu').pin_memory()
        G_diag = torch.zeros(K * D, device='cpu').pin_memory()

        # 预处理常量
        prior_mu = self.adapter.prior_mu.to(device).view(-1)
        prior_var = self.adapter.prior_var
        if prior_var.numel() == 1:  # 标量先验方差（各向同性）
            prior_prec = torch.full((K * D,), 1.0 / prior_var.item(), device=device)
        else:  # 向量或矩阵先验方差
            prior_prec = 1.0 / prior_var.view(-1).to(device)  # 强制展平

        print(f"Prior precision range: {prior_prec.min().item():.4f} - {prior_prec.max().item():.4f}")

        # 在外部定义 eye_K，避免内部函数未定义错误
        eye_K = torch.eye(K, device=device)

        def process_batch(x_batch, y_batch):
            with torch.no_grad():
                logits = self.forward_lp(x_batch)
                probs = torch.softmax(logits, dim=1)  # [B, K]

                # 更高效地计算完整的Hessian矩阵
                probs_unsq = probs.unsqueeze(2)  # [B, K, 1]
                Hess = eye_K.unsqueeze(0) * probs_unsq - torch.bmm(probs_unsq, probs_unsq.transpose(1, 2))  # [B, K, K]

                # 使用半精度减少内存占用
                Hess = Hess.half() if device.type == 'cuda' else Hess

                for i in range(x_batch.shape[0]):
                    x_i = x_batch[i:i + 1]
                    y_i = y_batch[i]
                    hess_i = Hess[i]  # [K, K] 完整的Hessian矩阵

                    # 计算残差项
                    y_onehot = F.one_hot(y_i, num_classes=K).float().to(device)
                    r_i = (probs[i] - y_onehot).cpu()  # residual, [K]

                    def f(proto):
                        return self.forward_lp_with_proto(x_i, proto).squeeze(0)  # [K]

                    # 计算Jacobian
                    J = jacobian(f, self.adapter.prototypes, create_graph=False, strict=False)  # [K, K, D]
                    J = J.reshape(K, -1).T.cpu()  # [K*D, K]

                    # 使用半精度减少内存
                    J_half = J.half() if device.type == 'cuda' else J
                    hess_i_half = hess_i.cpu().half() if device.type == 'cuda' else hess_i.cpu()

                    # 计算 J^T H
                    JH = J_half @ hess_i_half  # [K*D, K]

                    # 计算 diag(J^T H J) = sum_{i,j} J_{m,i} H_{i,j} J_{m,j}
                    contrib_diag = torch.sum(JH * J_half, dim=1).float()  # 转回float32 [K*D]
                    GGN_diag.add_(contrib_diag)

                    # 计算一阶梯度项：J^T * r
                    g_i = J @ r_i.unsqueeze(-1)  # [K*D, 1]
                    G_diag.add_(g_i.view(-1))

        # 分批处理
        total_batches = len(range(0, len(feature_train), batch_size))
        for b_idx, b in enumerate(tqdm(range(0, len(feature_train), batch_size), desc="Computing Diagonal GGN")):
            x_batch = feature_train[b:b + batch_size]
            y_batch = label_train[b:b + batch_size]
            process_batch(x_batch, y_batch)

            if b_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Batch {b_idx}/{total_batches} - GGN_diag mean: {GGN_diag.mean().item():.6f}")

        # 移动到GPU进行后续计算
        GGN_diag = GGN_diag.to(device)
        G_diag = G_diag.to(device)

        # 诊断信息
        print(f"GGN_diag min: {GGN_diag.min().item():.6f}, max: {GGN_diag.max().item():.6f}, mean: {GGN_diag.mean().item():.6f}")
        print(f"G_diag min: {G_diag.min().item():.6f}, max: {G_diag.max().item():.6f}, mean: {G_diag.mean().item():.6f}")


        # 计算协方差 = 1 / 精度
        cov_diag = 1 / (GGN_diag + prior_prec + 1e-6)

        # 后验均值 μ = Σ (G + Hθ* + λμ₀)
        theta_map = self.adapter.prototypes.flatten().to(device)
        term_g_raw     = G_diag
        term_map_raw   = GGN_diag * theta_map
        term_prior_raw = prior_prec * prior_mu
        b = term_g_raw + term_map_raw + term_prior_raw

        # 后验均值三项分解（乘协方差后）
        term_g_mu     = cov_diag * term_g_raw
        term_map_mu   = cov_diag * term_map_raw
        term_prior_mu = cov_diag * term_prior_raw
        g_weight = 1
        map_weight = 1
        prior_weight = 1
        posterior_mu  = g_weight * term_g_mu + map_weight * term_map_mu + prior_weight * term_prior_mu

        # 保存结果
        self.adapter.prototypes_map = posterior_mu.view(K, D).clone()
        self.adapter.ggn_cov = cov_diag.view(K, D)

        # ========= 协方差分析 =========
        print("\n=== 协方差分析 ===")
        print(f"GGN_diag range: {GGN_diag.min().item():.6f} - {GGN_diag.max().item():.6f}")
        print(f"Prior_prec range: {prior_prec.min().item():.6f} - {prior_prec.max().item():.6f}")
        print(f"Cov_diag range: {cov_diag.min().item():.6f} - {cov_diag.max().item():.6f}")
        print(f"Posterior_mu norm: {posterior_mu.norm().item():.4f}")

        if save_path:
            torch.save({
                'ggn_diag': self.adapter.ggn_cov.cpu(),
                'prototypes_map': self.adapter.prototypes_map.cpu()
            }, save_path)
            print(f"💾 Saved to {save_path}")

        # 内存清理
        del GGN_diag, G_diag, cov_diag, posterior_mu, b
        torch.cuda.empty_cache()
        gc.collect()

        # 检查后验参数
        print("\n检查后验参数--Prototypes map:")
        print(self.adapter.prototypes_map[:2, :5])
        print("GGN covariance (diag):")
        print(self.adapter.ggn_cov[:2, :5])

        return self.adapter.ggn_cov

    def compute_ggn_diag_covariance(self, feature_train, label_train, batch_size=16, save_path=None):
        import gc
        from torch.autograd.functional import jacobian
        from tqdm import tqdm
        import torch.nn.functional as F
        import torch

        # 计算后验前检查原型
        print("计算后验前检查原型--Prototypes before posterior calculation:")
        print(self.adapter.prototypes.data[:2, :5])

        # 开启 prototype 参数梯度
        self.adapter.prototypes.requires_grad_(True)
        K, D = self.adapter.prototypes.shape
        device = feature_train.device

        # 直接移动到 device
        feature_train = feature_train.to(device, non_blocking=True)
        label_train = label_train.to(device, non_blocking=True)

        # 初始化 GGN 对角项 和 一阶梯度项
        GGN_diag = torch.zeros(K * D, device='cpu').pin_memory()
        G_diag = torch.zeros(K * D, device='cpu').pin_memory()

        # 预处理常量
        prior_mu = self.adapter.prior_mu.to(device).view(-1)
        prior_var = self.adapter.prior_var
        if prior_var.numel() == 1:  # 标量先验方差（各向同性）
            prior_prec = torch.full((K * D,), 1.0 / prior_var.item(), device=device)
        else:  # 向量或矩阵先验方差
            prior_prec = 1.0 / prior_var.view(-1).to(device)  # 强制展平

        print(f"Prior precision range: {prior_prec.min().item():.4f} - {prior_prec.max().item():.4f}")

        # 在外部定义 eye_K，避免内部函数未定义错误
        eye_K = torch.eye(K, device=device)

        def process_batch(x_batch, y_batch):
            with torch.no_grad():
                logits = self.forward_lp(x_batch)
                probs = torch.softmax(logits, dim=1)  # [B, K]

                # 更高效地计算完整的Hessian矩阵
                probs_unsq = probs.unsqueeze(2)  # [B, K, 1]
                Hess = eye_K.unsqueeze(0) * probs_unsq - torch.bmm(probs_unsq, probs_unsq.transpose(1, 2))  # [B, K, K]

                # 使用半精度减少内存占用
                Hess = Hess.half() if device.type == 'cuda' else Hess

                for i in range(x_batch.shape[0]):
                    x_i = x_batch[i:i + 1]
                    y_i = y_batch[i]
                    hess_i = Hess[i]  # [K, K] 完整的Hessian矩阵

                    # 计算残差项
                    y_onehot = F.one_hot(y_i, num_classes=K).float().to(device)
                    r_i = (probs[i] - y_onehot).cpu()  # residual, [K]

                    def f(proto):
                        return self.forward_lp_with_proto(x_i, proto).squeeze(0)  # [K]

                    # 计算Jacobian
                    J = jacobian(f, self.adapter.prototypes, create_graph=False, strict=False)  # [K, K, D]
                    J = J.reshape(K, -1).T.cpu()  # [K*D, K]

                    # 使用半精度减少内存
                    J_half = J.half() if device.type == 'cuda' else J
                    hess_i_half = hess_i.cpu().half() if device.type == 'cuda' else hess_i.cpu()

                    # 计算 J^T H
                    JH = J_half @ hess_i_half  # [K*D, K]

                    # 计算 diag(J^T H J) = sum_{i,j} J_{m,i} H_{i,j} J_{m,j}
                    contrib_diag = torch.sum(JH * J_half, dim=1).float()  # 转回float32 [K*D]
                    GGN_diag.add_(contrib_diag)

                    # 计算一阶梯度项：J^T * r
                    g_i = J @ r_i.unsqueeze(-1)  # [K*D, 1]
                    G_diag.add_(g_i.view(-1))

        # 分批处理
        total_batches = len(range(0, len(feature_train), batch_size))
        for b_idx, b in enumerate(tqdm(range(0, len(feature_train), batch_size), desc="Computing Diagonal GGN")):
            x_batch = feature_train[b:b + batch_size]
            y_batch = label_train[b:b + batch_size]
            process_batch(x_batch, y_batch)

            if b_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Batch {b_idx}/{total_batches} - GGN_diag mean: {GGN_diag.mean().item():.6f}")

        # 移动到GPU进行后续计算
        GGN_diag = GGN_diag.to(device)
        G_diag = G_diag.to(device)

        # 诊断信息
        print(f"GGN_diag min: {GGN_diag.min().item():.6f}, max: {GGN_diag.max().item():.6f}, mean: {GGN_diag.mean().item():.6f}")
        print(f"G_diag min: {G_diag.min().item():.6f}, max: {G_diag.max().item():.6f}, mean: {G_diag.mean().item():.6f}")

        # 1. 对GGN_diag进行截断处理（防止极端值）
        ggn_upper = torch.quantile(GGN_diag, 0.95)  # 取95%分位数作为上限
        GGN_diag_clipped = GGN_diag.clamp(max=ggn_upper)

        # 2. 计算协方差（加入更强的数值稳定性保护）
        cov_diag = 1 / (GGN_diag_clipped + prior_prec + 1e-6)

        # 3. 获取当前MAP参数
        theta_map = self.adapter.prototypes.flatten().to(device)
        print(f"Theta MAP norm before update: {theta_map.norm().item():.4f}")

        # 4. 平衡三项量级（更鲁棒的标准化方式）
        # 梯度项标准化
        g_mean, g_std = G_diag.mean(), G_diag.std()
        term_g = (G_diag - g_mean) / (g_std + 1e-6)

        # GGN项标准化（使用对数缩放处理极端值）
        ggn_log = torch.log1p(GGN_diag_clipped.abs())
        ggn_scale = ggn_log.mean() + 1e-6
        term_ggn = (GGN_diag_clipped * theta_map) / ggn_scale

        # 先验项调整（动态匹配数据项量级）
        prior_scale = torch.abs(term_g).mean() / (prior_prec.mean() + 1e-6)
        term_prior = prior_scale * prior_prec * prior_mu

        # 5. 计算后验均值（加入阻尼因子防止过大的更新）
        damping_factor = 0.1  # 控制更新幅度
        posterior_mu = theta_map + damping_factor * cov_diag * (term_g + term_ggn + term_prior)

        # 保存结果
        self.adapter.prototypes_map = posterior_mu.view(K, D).clone()
        self.adapter.ggn_cov = cov_diag.view(K, D)

        # ========= 诊断分析 =========
        print("\n=== 更新量分析 ===")
        print(f"Theta MAP norm before: {theta_map.norm().item():.4f}")
        print(f"Posterior mu norm: {posterior_mu.norm().item():.4f}")
        print(f"Relative change: {(posterior_mu - theta_map).norm().item() / theta_map.norm().item():.4f}")

        print("\n=== 协方差分析 ===")
        print(f"GGN_diag clipped range: {GGN_diag_clipped.min().item():.6f} - {GGN_diag_clipped.max().item():.6f}")
        print(f"Cov_diag range: {cov_diag.min().item():.6f} - {cov_diag.max().item():.6f}")

        if save_path:
            torch.save({
                'ggn_diag': self.adapter.ggn_cov.cpu(),
                'prototypes_map': self.adapter.prototypes_map.cpu()
            }, save_path)
            print(f"💾 Saved to {save_path}")

        # 内存清理
        del GGN_diag, G_diag, cov_diag, posterior_mu
        torch.cuda.empty_cache()
        gc.collect()

        # 检查后验参数
        print("\n检查后验参数--Prototypes map:")
        print(self.adapter.prototypes_map[:2, :5])
        print("GGN covariance (diag):")
        print(self.adapter.ggn_cov[:2, :5])

        return self.adapter.ggn_cov

    def load_ggn_covariance(self, load_path, device='cuda:0'):
        checkpoint = torch.load(load_path, map_location='cpu')
        # self.adapter.ggn_cov = checkpoint['ggn_cov']
        self.adapter.ggn_cov = checkpoint['ggn_diag']
        self.adapter.prototypes_map = checkpoint['prototypes_map']
        if device is not None:
            self.adapter.ggn_cov = self.adapter.ggn_cov.to(device)
            self.adapter.prototypes_map = self.adapter.prototypes_map.to(device)
        print(f"Loaded GGN covariance and prototypes_map from {load_path}")

    def forward(self, image, return_features=False):

        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except:
            image_features = self.image_encoder(image.float())

        if "TR" in self.adapter.initialization:
            logits = self.forward_task_residual(image_features)
        elif "ClipA" in self.adapter.initialization:
            logits = self.forward_clipadapter(image_features)
        elif "TipA" in self.adapter.initialization:
            logits = self.forward_tipadapter(image_features)
        else:
            logits = self.forward_lp(image_features)

        if return_features:
            return logits, image_features
        else:
            return logits

    def forward_features(self, features):

        if "TR" in self.adapter.initialization:
            logits = self.forward_task_residual(features)
        elif "ClipA" in self.adapter.initialization:
            logits = self.forward_clipadapter(features)
        elif "TipA" in self.adapter.initialization:
            logits = self.forward_tipadapter(features)
        else:
            logits = self.forward_lp(features)

        return logits

    def forward_lp(self, features):
        device = features.device

        # 强制 adapter.prototypes 放到 device 上
        self.adapter.prototypes.data = self.adapter.prototypes.data.to(device)

        # 获取权重
        prototypes = self.adapter.prototypes

        # Normalize
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        # logit_scale 也要放到 device
        logit_scale = self.logit_scale.exp().to(device)

        logits = image_features_norm @ prototypes_norm.t() * logit_scale
        return logits

    def forward_lp_with_proto(self, features, proto):
        device = features.device

        # Normalize
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        proto_norm = proto / proto.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp().to(device)

        logits = image_features_norm @ proto_norm.t() * logit_scale
        return logits

    def forward_bays(self, features):
        """贝叶斯预测（MC采样）"""
        # 直接使用传入的特征（已预提取）

        features_norm = features / features.norm(dim=-1, keepdim=True).to('cuda:0')

        # 采样原型 [n_samples, K, D]
        prototypes_samples = self.adapter.sample_prototypes(self.adapter.n_samples).to('cuda:0')
        prototypes_norm = prototypes_samples / prototypes_samples.norm(dim=-1, keepdim=True)

        # 计算所有采样的logits [n_samples, B, K]
        logit_scale = self.logit_scale.exp().to('cuda:0')
        all_logits = torch.einsum('bd,skd->sbk', features_norm, prototypes_norm) * logit_scale

        # 计算不确定性（1 - max(softmax)）
        probs = F.softmax(all_logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # [n_samples, B]
        uncertainty = 1 - max_probs.mean(dim=0)  # 平均所有MC样本的不确定性

        # 返回均值预测和不确定性
        logits_mean = all_logits.mean(dim=0)
        return logits_mean, uncertainty


class TrainerXCostume(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        # Eval mode - not updating batchnorm statistics
        self.set_model_mode("eval")

        # Init kpis tracker
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Set number of batches to sample
        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size

        # Set features
        features = self.features_train.clone().cpu().numpy()
        labels = self.labels_train.clone()

        # Sample half dataset - to tackle previous oversample with text prompts
        if "CrossModal" in self.model.adapter.initialization:
            idx = np.random.choice(list(np.arange(0, features.shape[0])), features.shape[0] // 2)
            features = features[idx, :]
            labels = labels[idx]

        # Randomly shuffle
        idx = np.random.rand(features.shape[0]).argsort(axis=0)
        features = features[idx, :]
        labels = labels[idx]

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_init = self.batch_idx * self.batch_size
            batch_end = (self.batch_idx + 1) * self.batch_size

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(features[batch_init:batch_end],
                                                 labels[batch_init:batch_end])
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"{losses}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        return loss_summary


@TRAINER_REGISTRY.register()
class LaplaceBayesADAPTER(TrainerXCostume):
    """General Adapter
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LaplaceBayesADAPTER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LaplaceBayesADAPTER.PREC == "fp32" or cfg.TRAINER.LaplaceBayesADAPTER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()
        # NOTE: only give adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter", self.model.adapter, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LaplaceBayesADAPTER.PREC == "amp" else None

    def train(self):
        # 初始化记录最佳模型的属性
        self.best_acc = 0.0
        self.best_model_state = None

        self.set_model_mode("eval")

        # Feature extraction on test set
        self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
        print("Zero-Shot accuracy on test: " +
              str(round(compute_accuracy(output_test.cuda(), self.labels_test.cuda())[0].item(), 2)))

        # Feature extraction on training set
        self.labels_train, self.logits_zs, self.features_train = self.extract_features(
            partition="train", reps=self.model.adapter.epochs_aumentation, transforms=self.model.adapter.augmentations)


        # Init alphas in constraint formulation
        if self.model.adapter.apply_constraint != "none":
            print("Getting initial lagrangian multipliers for constraint formulation", end="\n")
            self.model.adapter.device = self.device
            self.model.adapter.init_lagrangian_multipliers(self.labels_train, self.logits_zs)
            print("Lagrangian multipliers: ")
            print(list(torch.round(self.model.adapter.alpha_constraint.detach(), decimals=3).cpu().numpy()))

        # Training of adapter
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):

            # Train and update weights per epoch
            self.before_epoch()
            loss_summary = self.run_epoch()

            if loss_summary["acc_test"] > self.best_acc:
                self.best_acc = loss_summary["acc_test"]
                self.best_model_state = copy.deepcopy(self.model.adapter.state_dict())
                print(f"Best acc updated: {self.best_acc:.4f}")

            # Update lagrangian parameter and multiplier
            if "adaptative" in self.model.adapter.apply_constraint:
                self.model.adapter.outer_step()

            self.after_epoch()

        ggn_dir = osp.join(self.cfg.OUTPUT_DIR, "ggn_checkpoints")
        os.makedirs(ggn_dir, exist_ok=True)
        ggn_path = osp.join(ggn_dir, f"ggn_cov_seed{self.cfg.SEED}.pt")  # 注意：SEED 也可能是 cfg.SEED

        self.after_train(self.features_train, self.labels_train,
                         self.features_test, self.labels_test,
                         ggn_save_path=None)  # 保存路径改成ggn_path就可以保存调试

    def bayesian_evaluation(self, feature_test, label_test, mode="linearized", posterior_path=None):
        self.set_model_mode("eval")
        # 自动加载后验参数
        if mode == "linearized":
            if (self.model.adapter.prototypes_map is None or
                    self.model.adapter.ggn_cov is None):
                assert posterior_path is not None, "未提供后验路径，且未预加载 prototypes_map / ggn_cov。"
                self.model.load_ggn_covariance(posterior_path, device=self.device)
        features = feature_test.to(self.device)
        labels = label_test.to(self.device)

        acc,ece,aece = self.linearized_predict_batchwise(features,labels,n_samples=self.model.adapter.n_samples,batch_size= 64,T=1)
        # final_result = self.search_optimal_temperature(features, labels,
        #                                                                               n_samples=self.model.adapter.n_samples,
        #                                                                               batch_size=64,
        #                                                                               T_range=[1])
        # print("\n=== 最优结果汇总 ===")
        # print(f"1. ECE最优温度: {final_result['best_T_for_ece']:.2f}")
        # print(f"   - 对应ACC: {final_result['best_acc_at_best_ece'] * 100:.2f}%")
        # print(f"   - 对应ECE: {final_result['best_ece']:.4f}")
        #
        # print(f"\n2. ACC最优温度: {final_result['best_T_for_acc']:.2f}")
        # print(f"   - 对应ACC: {final_result['best_acc'] * 100:.2f}%")
        # print(f"   - 对应ECE: {final_result['best_ece_at_best_acc']:.4f}")

    def _compute_ece(self, confidences, preds, labels, n_bins=10, save_csv_path=None):
        """计算 ECE 和 AECE，同时支持输出 CSV 文件保存 bin-wise 结果"""

        import pandas as pd

        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidences.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        aece = 0.0
        total_samples = len(confidences)

        print(f"{'Bin':<10}{'Count':<10}{'Acc':<10}{'Conf':<10}{'|Gap|':<10}")
        print("-" * 50)

        bin_data = []

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_count = in_bin.sum().item()

            if bin_count > 0:
                bin_acc = (preds[in_bin] == labels[in_bin]).float().mean().item()
                bin_conf = confidences[in_bin].mean().item()
                abs_gap = abs(bin_acc - bin_conf)
                bin_prob = bin_count / total_samples

                ece += abs_gap * bin_prob
                aece += abs_gap / n_bins

                print(f"{i + 1:<10}{bin_count:<10}{bin_acc:<10.4f}{bin_conf:<10.4f}{abs_gap:<10.4f}")
            else:
                bin_acc, bin_conf, abs_gap = 'NA', 'NA', 'NA'
                print(f"{i + 1:<10}{0:<10}{bin_acc:<10}{bin_conf:<10}{abs_gap:<10}")

            # 保存一行 bin 数据
            bin_data.append({
                'Bin': i + 1,
                'Bin Range': f"({bin_lower:.1f}, {bin_upper:.1f}]",
                'Count': bin_count,
                'Accuracy': bin_acc,
                'Confidence': bin_conf,
                'Gap': abs_gap
            })

        print("-" * 50)
        print(f"Final ECE:  {ece:.4f}")
        print(f"Final AECE: {aece:.4f}")

        # 加入 summary 行
        bin_data.append({
            'Bin': 'Summary',
            'Bin Range': '',
            'Count': total_samples,
            'Accuracy': f"Overall Acc = {(preds == labels).float().mean().item():.4f}",
            'Confidence': f"ECE = {ece:.4f}",
            'Gap': f"AECE = {aece:.4f}"
        })

        # ✅ 可选保存 CSV
        if save_csv_path is not None:
            df = pd.DataFrame(bin_data)
            df.to_csv(save_csv_path, index=False)
            print(f"[Saved bin-wise ECE analysis to: {save_csv_path}]")

        return ece, aece

    def _compute_adaptive_ece(self, confidences, predictions, labels, n_bins=10, save_csv_path=None):
        """
        同时计算自适应分箱(AECE)和传统分箱(ECE)
        AECE: 等样本数分箱
        ECE: 等置信度间隔分箱
        返回: (ece, aece, overall_acc)
        """
        import pandas as pd
        import numpy as np

        # 转换为numpy数组
        confidences = confidences.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # 计算整体准确率
        overall_acc = np.mean(predictions == labels)
        overall_conf = np.mean(confidences)

        # 1. 自适应分箱 (AECE) - 等样本数分箱
        sorted_indices = np.argsort(confidences)
        conf_sorted = confidences[sorted_indices]
        pred_sorted = predictions[sorted_indices]
        label_sorted = labels[sorted_indices]

        total = len(conf_sorted)
        bin_size = total // n_bins
        ece = 0.0
        aece_stats = []

        for i in range(n_bins):
            start = i * bin_size
            end = total if i == n_bins - 1 else (i + 1) * bin_size
            if end <= start:
                continue

            bin_conf = conf_sorted[start:end]
            bin_pred = pred_sorted[start:end]
            bin_label = label_sorted[start:end]

            acc = np.mean(bin_pred == bin_label)
            conf = np.mean(bin_conf)
            gap = abs(acc - conf)
            ece += gap * len(bin_conf) / total  # ECE计算

            aece_stats.append([i + 1, len(bin_conf), acc, conf, gap])

        aece = np.mean([x[-1] for x in aece_stats])  # AECE计算

        # 添加整体统计到AECE
        aece_stats.append([0, total, overall_acc, overall_conf, abs(overall_acc - overall_conf)])

        # 2. 传统分箱 (ECE) - 等置信度间隔分箱
        ece_stats = []
        bin_edges = np.linspace(0, 1, n_bins + 1)
        total_ece = 0.0

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i+1]
            in_bin = (confidences >= low) & (confidences < high)
            # 处理最后一个区间包含1.0
            if i == n_bins - 1:
                in_bin = (confidences >= low) & (confidences <= high)

            if not np.any(in_bin):
                continue

            bin_conf = confidences[in_bin]
            bin_pred = predictions[in_bin]
            bin_label = labels[in_bin]

            acc = np.mean(bin_pred == bin_label)
            conf = np.mean(bin_conf)
            gap = abs(acc - conf)
            total_ece += gap * len(bin_conf) / total  # ECE计算

            ece_stats.append([i + 1, len(bin_conf), low, high, acc, conf, gap])

        # 添加整体统计到ECE
        ece_stats.append([0, total, 0.0, 1.0, overall_acc, overall_conf, abs(overall_acc - overall_conf)])

        # 保存CSV文件
        if save_csv_path is not None:
            # 自适应分箱(AECE)结果
            aece_df = pd.DataFrame(
                aece_stats,
                columns=["Bin", "Count", "Acc", "Conf", "|Gap|"]
            )
            aece_path = save_csv_path.replace(".csv", "_aece.csv")
            aece_df.to_csv(aece_path, index=False)
            print(f"[Saved adaptive ECE (AECE) bin stats to {aece_path}]")

            # 传统分箱(ECE)结果
            ece_df = pd.DataFrame(
                ece_stats,
                columns=["Bin", "Count", "Low", "High", "Acc", "Conf", "|Gap|"]
            )
            ece_path = save_csv_path.replace(".csv", "_ece.csv")
            ece_df.to_csv(ece_path, index=False)
            print(f"[Saved traditional ECE bin stats to {ece_path}]")

        return ece, aece, overall_acc

    def linearized_predict_batchwise(
            self,
            features,
            labels,
            n_samples=1000,
            batch_size=16,
            T=1.0,  # 🔥 温度缩放
    ):
        """
        分 batch 做贝叶斯预测，但统一收集 logits + labels，在最后统一计算 Accuracy 和 ECE。
        更精确评估模型校准。
        """
        assert self.model.adapter.prototypes_map is not None, "必须先计算后验分布!"

        K, D = self.model.adapter.prototypes_map.shape
        device = features.device
        features = features.to(device)
        labels = labels.to(device)


        theta_map = self.model.adapter.prototypes.flatten().to(device)
        theta_samples = self._sample_from_posterior_diag(n_samples).to(device)  # [n_samples, K*D]
        # theta_samples = self._sample_from_posterior(n_samples).to(device)  # [n_samples, K*D]

        probs_all = []  # 改为直接存储概率
        labels_all = []

        total = features.size(0)
        for start in tqdm(range(0, total, batch_size), desc="Batchwise Bayesian Predict"):
            end = min(start + batch_size, total)
            sub_features = features[start:end]
            sub_labels = labels[start:end]

            logits_for_grad = self.model.forward_lp(sub_features)
            logits_map = logits_for_grad.detach()

            probs_samples_batch = []

            for i in range(end - start):
                J_i = []
                for k in range(K):
                    grad_k = torch.autograd.grad(
                        logits_for_grad[i, k],
                        self.model.adapter.prototypes,
                        retain_graph=True,
                        create_graph=False
                    )[0].flatten().detach()
                    J_i.append(grad_k)
                J_i = torch.stack(J_i, dim=0)  # [K, K*D]

                logits_i_samples = []
                for s in range(n_samples):
                    delta = theta_samples[s] - theta_map  # [K*D]
                    correction = (J_i @ delta.unsqueeze(-1)).reshape(K)
                    logits_i = logits_map[i] + correction
                    logits_i_samples.append(logits_i)
                logits_i_samples = torch.stack(logits_i_samples)  # [n_samples, K]
                probs_i_samples = torch.softmax(logits_i_samples / T, dim=1)  # [n_samples, K]
                probs_samples_batch.append(probs_i_samples)

            # 堆叠当前批次所有样本的概率 [n_samples, B, K]
            probs_samples_batch = torch.stack(probs_samples_batch, dim=1)
            probs_mean = probs_samples_batch.mean(dim=0)  # [B, K]

            probs_all.append(probs_mean.detach().cpu())
            labels_all.append(sub_labels.detach().cpu())

            del logits_for_grad, logits_map, probs_samples_batch
            if start % (10 * batch_size) == 0:  # 每10批清理一次
                torch.cuda.empty_cache()
                gc.collect()

        # === 拼接所有概率和标签 ===
        probs_all = torch.cat(probs_all, dim=0)  # [B, K]
        labels_all = torch.cat(labels_all, dim=0)  # [B]

        preds = probs_all.argmax(dim=1)
        acc = (preds == labels_all).float().mean().item()
        confidences = probs_all.max(dim=1).values

        # === 计算校准指标 ===
        ece, aece, _ = self._compute_adaptive_ece(
            confidences, preds, labels_all, n_bins=10,
            save_csv_path=f"{self.cfg.OUTPUT_DIR}/calibration_results.csv"
        )

        print(f"\nFinal Bayesian Evaluation with T={T}:")
        print(f"Accuracy:      {acc * 100:.2f}%")
        print(f"Avg Confidence:{confidences.mean().item():.4f}")
        print(f"ECE:           {ece:.4f}")
        print(f"AECE:          {aece:.4f}")

        return acc, ece, aece
    
    def linearized_predict_batchwise_nonlinear(
            self,
            features,
            labels,
            n_samples=1000,
            batch_size=16,
            T=1.0,  # 🔥 温度缩放
    ):
        """
        分 batch 做贝叶斯预测，直接从后验采样参数并平均 logits。
        统一收集 logits + labels，最后计算 Accuracy 和 ECE。
        """
        assert self.model.adapter.prototypes_map is not None, "必须先计算后验分布!"
        print('非线性预测，直接进行贝叶斯预测（Lplace-ggn）')

        K, D = self.model.adapter.prototypes_map.shape
        device = features.device
        features = features.to(device)
        labels = labels.to(device)

        # 从后验采样参数（直接使用采样参数计算 logits，无需线性修正）
        theta_samples = self._sample_from_posterior_diag(n_samples).to(device)  # [n_samples, K*D]

        logits_all = []
        labels_all = []

        # 保存原始 prototypes 的值
        original_prototypes = self.model.adapter.prototypes.data.clone()

        total = features.size(0)
        for start in tqdm(range(0, total, batch_size), desc="Batchwise Bayesian Predict"):
            end = min(start + batch_size, total)
            sub_features = features[start:end]
            sub_labels = labels[start:end]

            # 直接对每个采样参数计算 logits（无需梯度计算）
            logits_samples = []
            for s in range(n_samples):
                # 直接修改 prototypes 的值（不改变 Parameter 类型）
                with torch.no_grad():
                    self.model.adapter.prototypes.data = theta_samples[s].reshape(K, D).to(device)
                # 计算当前采样下的 logits
                logits_s = self.model.forward_lp(sub_features).detach()  # [batch_size, K]
                logits_samples.append(logits_s)

            logits_samples = torch.stack(logits_samples, dim=0)  # [n_samples, batch_size, K]
            logits_mean = logits_samples.mean(dim=0)  # [batch_size, K]
            logits_scaled = logits_mean / T  # 温度缩放

            logits_all.append(logits_scaled.cpu())
            labels_all.append(sub_labels.cpu())

            # 恢复原始 prototypes 的值
            with torch.no_grad():
                self.model.adapter.prototypes.data = original_prototypes.to(device)
            torch.cuda.empty_cache()

        # 拼接所有结果
        logits_all = torch.cat(logits_all, dim=0)  # [B, K]
        labels_all = torch.cat(labels_all, dim=0)  # [B]

        # 计算指标
        probs = torch.softmax(logits_all, dim=1)
        preds = probs.argmax(dim=1)
        acc = (preds == labels_all).float().mean().item()
        confidences = probs.max(dim=1).values

        ece, aece, _ = self._compute_adaptive_ece(
            confidences, preds, labels_all, n_bins=10,
            save_csv_path=f"{self.cfg.OUTPUT_DIR}/calibration_results.csv"
        )

        print(f"\nFinal Bayesian Evaluation with T={T}:")
        print(f"Accuracy:      {acc * 100:.2f}%")
        print(f"Avg Confidence:{confidences.mean().item():.4f}")
        print(f"ECE:           {ece:.4f}")
        print(f"AECE:          {aece:.4f}")

        return acc, ece, aece


    def search_optimal_temperature(self, features, labels, n_samples=500, batch_size=32, T_range=None, bins=10):
        """
        自动搜索最优温度 T，同时记录:
        1. 最小ECE对应的T和ACC
        2. 最大ACC对应的T和ECE
        """
        from collections import defaultdict

        if T_range is None:
            T_range = np.arange(0.6, 1.41, 0.05)

        # 初始化记录变量
        best_T_for_ece = None
        best_ece = float("inf")
        best_acc_at_best_ece = None

        best_T_for_acc = None
        best_acc = -float("inf")
        best_ece_at_best_acc = None

        results = defaultdict(list)

        print("\n🔍 Searching optimal temperature for calibration...")
        for T in T_range:
            acc, ece, _ = self.linearized_predict_batchwise(
                features, labels,
                n_samples=n_samples,
                batch_size=batch_size,
                T=T,
                debug=False
            )

            # 记录所有结果
            print(f"  T = {T:.2f} | Acc = {acc * 100:.2f}%, ECE = {ece:.4f}")
            results["T"].append(T)
            results["acc"].append(acc)
            results["ece"].append(ece)

            # 更新最优ECE记录
            if ece < best_ece:
                best_ece = ece
                best_T_for_ece = T
                best_acc_at_best_ece = acc

            # 更新最优ACC记录
            if acc > best_acc:
                best_acc = acc
                best_T_for_acc = T
                best_ece_at_best_acc = ece

        print("\n✅ Optimal Results:")
        print(
            f"[For Best ECE] T = {best_T_for_ece:.2f} | Acc = {best_acc_at_best_ece * 100:.2f}%, ECE = {best_ece:.4f}")
        print(
            f"[For Best ACC] T = {best_T_for_acc:.2f} | Acc = {best_acc * 100:.2f}%, ECE = {best_ece_at_best_acc:.4f}")

        return {
            'best_T_for_ece': best_T_for_ece,
            'best_acc_at_best_ece': best_acc_at_best_ece,
            'best_ece': best_ece,
            'best_T_for_acc': best_T_for_acc,
            'best_acc': best_acc,
            'best_ece_at_best_acc': best_ece_at_best_acc,
            'all_results': results
        }

    def _sample_from_posterior(self, n_samples):
        """从后验高斯分布采样参数"""
        K, D = self.model.adapter.prototypes_map.shape
        mu = self.model.adapter.prototypes_map.flatten().to('cuda:0')  # [K*D]
        cov = self.model.adapter.ggn_cov.reshape(K * D, K * D).to('cuda:0')  # [K*D, K*D]

        # ✅ 首次尝试 Cholesky 分解（不添加正则）
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            print("⚠️ Covariance not positive-definite. Adding small jitter (1e-6 * I).")
            cov = cov + 1e-6 * torch.eye(K * D, device='cuda:0')
            L = torch.linalg.cholesky(cov)

        # ✅ 使用分解结果进行采样
        eps = torch.randn(n_samples, K * D, device='cuda:0')
        samples = mu + eps @ L.T  # [n_samples, K*D]

        return samples

    def _sample_from_posterior_diag(self, n_samples):
        """从对角协方差的后验高斯分布中采样参数"""
        K, D = self.model.adapter.prototypes_map.shape
        mu = self.model.adapter.prototypes_map.flatten().to('cuda:0')  # shape: (K*D,)
        var_diag = self.model.adapter.ggn_cov.flatten().to('cuda:0')  # shape: (K*D,)

        # 对角协方差采样只需要对每个维度加上正态扰动
        eps = torch.randn(n_samples, K * D, device='cuda:0')  # shape: (n_samples, K*D)
        std = torch.sqrt(var_diag).unsqueeze(0)  # shape: (1, K*D)
        samples = mu.unsqueeze(0) + eps * std  # shape: (n_samples, K*D)

        return samples  # [n_samples, K*D]

    def after_train(self, feature_train, label_train, feature_test, label_test, ggn_save_path=None):
        import os
        if self.best_model_state is not None:
            # 保存当前prior_var
            current_prior_var = self.model.adapter.prior_var.clone()

            self.model.adapter.load_state_dict(self.best_model_state)
            print(f"Loaded best model with acc: {self.best_acc:.4f}")

            # 恢复prior_var
            self.model.adapter.prior_var = current_prior_var
            print(f"Restored prior_var to: {current_prior_var.mean().item():.4f}")

        if ggn_save_path is not None and os.path.exists(ggn_save_path):
            print("Loading saved Laplace-GGN covariance...")
            self.model.load_ggn_diag_covariance(ggn_save_path)
        else:
            print("Computing Laplace-GGN covariance...")
            self.model.compute_ggn_diag_covariance(feature_train, label_train, save_path=ggn_save_path)  # 不保存字典
            # self.model.compute_ggn_covariance(feature_train, label_train, save_path=ggn_save_path)  # 不保存字典

        print("Evaluating Bayesian predictions...")
        self.bayesian_evaluation(feature_test, label_test)

        print("Finish training")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def compute_confidence_penalty(self, logits, reduction='mean'):
        """
        计算 entropy 惩罚项（用于降低模型置信度，从而降低 ECE）
        避免模型过度自信
        Args:
            logits: Tensor[B, K]
            reduction: 'mean' or 'none'
        Returns:
            penalty: Tensor[1] or Tensor[B]
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)  # shape: [B]

        if reduction == 'mean':
            return entropy.mean()
        else:
            return entropy

    def forward_backward(self, features, labels):
        prec = self.cfg.TRAINER.LaplaceBayesADAPTER.PREC
        if prec == "amp":
            with autocast():
                # Cross-entropy loss
                output = self.model.forward_features(torch.tensor(features).to(self.device))
                # Softmax cross-entropy
                loss_ce = F.cross_entropy(output, labels)
                # Constraint to zero-shot (CLAP)
                if self.model.adapter.apply_constraint != "none":
                    loss_constraint = self.model.adapter.zero_shot_constraint()
                    loss = loss_ce + loss_constraint
                else:
                    loss = loss_ce
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # Cross-entropy loss
            output = self.model.forward_features(torch.tensor(features).to(self.device))
            # Softmax cross-entropy
            loss_ce = F.cross_entropy(output, labels)
            # entropy
            confidence_penalty = self.compute_confidence_penalty(output, reduction='mean')
            # 余弦退火 置信度惩罚项的权重系数
            lambda_init = 2.0
            lambda_final = 0.2
            lambda_conf = lambda_final + 0.5 * (lambda_init - lambda_final) * (
                        1 + math.cos(math.pi * self.epoch / self.max_epoch))
            # lambda_conf = 0.1
            # Constraint to zero-shot (CLAP)
            if self.model.adapter.apply_constraint != "none":
                loss_constraint = self.model.adapter.zero_shot_constraint()
                loss = loss_ce + loss_constraint + lambda_conf * confidence_penalty
                print('loss_ce:', loss_ce.item())
                print('loss_constraint:', loss_constraint.item())
                print('lambda_conf:',lambda_conf)
                print('lambda_conf * confidence_penalty:', lambda_conf * confidence_penalty.item())
            else:
                loss = loss_ce

            self.model_backward_and_update(loss)
        # 训练一个epoch 测试一次
        with torch.no_grad():
            output_test = self.model.forward_features(self.features_test.clone().detach().to(self.device))

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, labels)[0].item(),
            "acc_test": compute_accuracy(output_test, self.labels_test)[0].item(),
        }

        # 每个epoch后更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # 打印每个param_group的学习率
            for i, param_group in enumerate(self.optim.param_groups):
                print(
                    f"Epoch {self.epoch + 1} Batch {self.batch_idx + 1} - param_group {i} learning rate: {param_group['lr']}")

        torch.cuda.empty_cache()
        return loss_summary

    def load_model(self, directory, cfg, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        else:
            print("Pretrained model given")

        if self.model.adapter.initialization == "TipA":
            epoch = 1

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            else:
                print('Model found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            if "TipA" in self.model.adapter.initialization:
                self.model.adapter.cache_keys = nn.Parameter(state_dict['cache_keys'].clone())
                self.model.adapter.cache_values = nn.Parameter(state_dict['cache_values'].clone())

            if self.cfg.DATASET.NAME == 'ImageNetA' or self.cfg.DATASET.NAME == 'ImageNetR':
                if self.cfg.DATASET.NAME == 'ImageNetA':
                    from datasets.imagenet_a_r_indexes_v2 import find_imagenet_a_indexes as find_indexes
                else:
                    from datasets.imagenet_a_r_indexes_v2 import find_imagenet_r_indexes as find_indexes
                imageneta_indexes = find_indexes()
                print("Parameters found: ")
                print(state_dict.keys())
                state_dict['base_text_features'] = state_dict['base_text_features'][imageneta_indexes]
                state_dict['prototypes'] = state_dict['prototypes'][imageneta_indexes]

                if "TipA" in self.model.adapter.initialization:
                    state_dict['cache_values'] = state_dict['cache_values'][:, imageneta_indexes]
                    self.model.adapter.cache_keys = nn.Parameter(state_dict['cache_keys'].clone())
                    self.model.adapter.cache_values = nn.Parameter(state_dict['cache_values'].clone())
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            self.model.float()

    def extract_features(self, partition, reps=1, transforms=False):
        print("Extracting features from: " + partition, end="\n")
        self.set_model_mode("eval")

        if partition == "train":

            # Copy safe version of training dataloader
            data_loader = copy.deepcopy(self.train_loader_x)

            # Set data augmentation transforms
            if not transforms:
                data_loader.dataset.transform = self.val_loader.dataset.transform

            # Set data loader with drop last to false for not losing samples
            data_loader = torch.utils.data.DataLoader(
                copy.deepcopy(self.train_loader_x.dataset), batch_size=self.train_loader_x.batch_size,
                sampler=self.train_loader_x.sampler, num_workers=self.train_loader_x.num_workers,
                drop_last=False, pin_memory=self.train_loader_x.pin_memory)

        elif partition == "val":
            data_loader = copy.deepcopy(self.val_loader)
        elif partition == "test":
            data_loader = copy.deepcopy(self.test_loader)
        else:
            assert False

        if "TipA" not in self.model.adapter.initialization:

            labels_ds, logits_ds, features_ds = [], [], []
            for rep in range(reps):
                for batch_idx, batch in enumerate(tqdm(data_loader)):
                    with torch.no_grad():
                        input, label = self.parse_batch_test(batch)
                        logits, features = self.model(input, return_features=True)
                        labels_ds.append(label), logits_ds.append(logits.cpu()), features_ds.append(features.cpu())

            # Concatenate outputs
            labels_ds = torch.cat(labels_ds, dim=0)
            logits_ds = torch.cat(logits_ds, dim=0)
            features_ds = torch.cat(features_ds, dim=0)

        else:

            labels_ds, logits_ds, features_ds = [], [], []
            for rep in range(reps):
                labels_ds_irep, logits_dsirep, features_ds_irep = [], [], []
                for batch_idx, batch in enumerate(tqdm(data_loader)):
                    with torch.no_grad():
                        input, label = self.parse_batch_test(batch)
                        logits, features = self.model(input, return_features=True)
                        labels_ds_irep.append(label), logits_dsirep.append(logits.cpu()), features_ds_irep.append(
                            features.cpu())
                # Concatenate outputs for dataset
                labels_ds_irep = torch.cat(labels_ds_irep, dim=0)
                logits_dsirep = torch.cat(logits_dsirep, dim=0)
                features_ds_irep = torch.cat(features_ds_irep, dim=0)
                # Concatenate outputs for repetitions
                labels_ds.append(labels_ds_irep.unsqueeze(0))
                logits_ds.append(logits_dsirep.unsqueeze(0))
                features_ds.append(features_ds_irep.unsqueeze(0))

            # Concatenate outputs
            labels_ds = torch.cat(labels_ds, dim=0)[0, :]
            logits_ds = torch.cat(logits_ds, dim=0).mean(0)
            features_ds = torch.cat(features_ds, dim=0).mean(0)

        return labels_ds, logits_ds, features_ds
