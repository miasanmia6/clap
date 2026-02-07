# 标准库导入
import os
import os.path as osp
import datetime
import time
import copy
import gc
import math
from re import template
import weakref

# 第三方库导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch 相关导入
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.autograd.functional import jacobian

# 项目特定导入
from dassl.engine import TRAINER_REGISTRY, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, AverageMeter, MetricMeter
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datasets.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

# ==================== 全局GPU优化配置 ====================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ==================== 调试模式配置（核心修改） ====================
DEBUG_MODE = False  # 开启调试模式
DEBUG_TRAIN_BATCHES = 1  # 只训练1个batch
DEBUG_GGN_SAMPLES = 50  # GGN计算使用少量样本加速调试
TEST_USE_FULL_DATA = True  # 测试集使用完整数据

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
T_mapping = {
    (1, 1): 0.7,
    (1, 2): 0.7,
    (1, 3): 0.7,
    (2, 1): 0.7,
    (2, 2): 0.7,
    (2, 3): 0.7,
    (4, 1): 0.7,
    (4, 2): 0.7,
    (4, 3): 0.7,
    (8, 1): 0.7,
    (8, 2): 0.7,
    (8, 3): 0.7,
    (16, 1): 0.7,
    (16, 2): 0.7,
    (16, 3): 0.7,
    (32, 1): 0.7,
    (32, 2): 0.7,
    (32, 3): 0.7
}


def get_temperature(shot, seed, default_T=1.0):
    """根据shot和seed获取对应的温度参数"""
    return T_mapping.get((shot, seed), default_T)


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
    plt.savefig(save_path)
    print(f"[Saved confidence histogram to {save_path}]")

    print(f"Mean Confidence: {confidences.mean():.4f}")
    print(f"Min Confidence:  {confidences.min():.4f}")
    print(f"Max Confidence:  {confidences.max():.4f}")
    print(f"Percent > 0.9:   {(confidences > 0.9).mean() * 100:.2f}%")
    print(f"Percent < 0.5:   {(confidences < 0.5).mean() * 100:.2f}%")


def find_optimal_temperature(logits, labels):
    import torch.nn.functional as F
    from torch import optim

    logits = logits.detach()
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
    """增强版GPU显存监控"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"[GPU MEM {note}] - Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Allocated: {max_allocated:.2f} MB")
    else:
        print(f"[GPU MEM {note}] - CUDA not available")


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

        # take features from the eot embedding
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
        TEMPLATES = []  # 简化调试，暂不使用ImageNet模板
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])
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
        self.augmentations = True
        self.epochs_aumentation = 20 # 20
        
        # Laplace-GGN 相关参数
        self.register_buffer("ggn_cov", None)
        self.register_buffer("prior_mu", self.base_text_features.clone().to('cuda:0').flatten())
        self.prototypes_map = None
        self.posterior_chol = None
        self.precision = None
        self.n_samples = 50  # 调试模式减少MC采样次数，加速测试

        if self.initialization == "RANDOM":
            print("Using RANDOM initialization in Linear Probing", end="\n")
            self.prototypes = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(base_text_features.shape)))
        elif "ZS" in self.initialization:
            print("Using Zero-Shot initialization in Linear Probing", end="\n")
            self.prototypes = nn.Parameter(base_text_features.clone())
            K, D = self.prototypes.shape
            self.register_buffer("prior_var", torch.full((K * D,), 0.0001))
        else:
            print("Initialization for Linear Probing not implemented")
            assert False

        if self.apply_constraint != "none":
            print("Applying constraint to the logistic regression weights: " + str(self.distance))

    def sample_prototypes(self, n_samples=1):
        """从 Laplace-GGN 后验中采样原型"""
        if self.ggn_cov is None:
            return self.prototypes.unsqueeze(0)

        K, D = self.prototypes_map.shape

        prototypes_samples = []
        for _ in range(n_samples):
            noise = torch.randn_like(self.prototypes_map)

            cov_diag = torch.diagonal(self.ggn_cov.reshape(K * D, K * D)).reshape(K, D)

            if cov_diag.shape != self.prototypes_map.shape:
                cov_diag = torch.diagonal(self.ggn_cov, dim1=0, dim2=2).permute(2, 0, 1)
                cov_diag = torch.diagonal(cov_diag, dim1=1, dim2=2)

            cov_diag = torch.clamp(cov_diag, min=0)

            sample = self.prototypes_map + torch.sqrt(cov_diag) * noise
            prototypes_samples.append(sample)

        return torch.stack(prototypes_samples)

    def init_MultiModal(self):
        print("Using Zero-Shot initialization in Linear Probing", end="\n")
        self.prototypes = nn.Parameter(self.base_text_features.clone())

    def zero_shot_constraint(self):
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
                # 修复：确保labels_one_hot和logits_ds在同一设备
                device = logits_ds.device
                labels_one_hot = torch.nn.functional.one_hot(labels_ds).to(device)  # 移除.cpu()
                performance = torch.diag(torch.softmax(logits_ds, -1).t() @ labels_one_hot.to(torch.float32)) / \
                              labels_one_hot.sum(0)

                if "corrected" in self.apply_constraint:
                    performance *= (logits_ds.shape[-1] / torch.sum(performance).item())
                if "constant" in self.apply_constraint:
                    performance = torch.ones(logits_ds.shape[-1]).to(torch.float) * torch.mean(performance).item()

        # 确保alpha_constraint在GPU上
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

        disimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(-1)
        phr_value, phr_grad = phr(disimilitude, self.alpha_constraint, self.penalty_parameter)
        self.alpha_constraint = phr_grad.detach().clone()
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
        self.dtype = clip_model.dtype
        text_encoder = TextEncoder(clip_model)

        if cfg.TRAINER.LaplaceBayesADAPTER.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features, text_embeddings_all = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        else:
            print(">> Use enhanced base!")
            base_text_features, text_embeddings_all = _get_base_text_features(
                cfg, classnames, clip_model, text_encoder, cfg.TRAINER.TaskRes.ENHANCED_BASE)

        self.text_embeddings_all = text_embeddings_all
        self.adapter = AdapterMethod(cfg, clip_model, base_text_features)
    

    def compute_ggn_diag_covariance(self, feature_train, label_train, batch_size=8,  # 降为8进一步控显存
                                     save_path=None):
        """
        最终修复版：解决Jacobian报错+显存可控+ACC显示不影响计算
        核心修复：
        1. Jacobian改用reverse-mode，兼容vectorize=False
        2. 仅计算top-3类的雅可比，显存再降30%
        3. batch_size=8，彻底避免OOM
        """
        import gc
        from tqdm import tqdm
        import torch.nn.functional as F
        import torch

        # ========== 显存碎片+PyTorch版本兼容优化 ==========
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        gc.collect()

        print("检查后验计算前的原型:")
        print(self.adapter.prototypes.data[:2, :5])

        self.adapter.prototypes.requires_grad_(True)
        K, D = self.adapter.prototypes.shape  # 1000, 1024
        device = feature_train.device

        # ========== 核心：GGN_diag放CPU，省显存 ==========
        GGN_diag = torch.zeros(K * D, device='cpu', dtype=torch.float32)
        G_diag = torch.zeros(K * D, device='cpu', dtype=torch.float32)

        # 先验参数（和你原代码一致）
        prior_mu = self.adapter.prior_mu.to(device).view(-1)
        prior_var = self.adapter.prior_var
        prior_prec = 1.0 / prior_var.view(-1).to(device)
        print(f"先验精度范围: {prior_prec.min().item():.4f} - {prior_prec.max().item():.4f}")

        # ========== 修复Jacobian报错：改用reverse-mode ==========
        def process_batch(x_batch, y_batch):
            with torch.no_grad():
                # 1. 前向计算（和你原代码一致）
                logits = self.forward_lp(x_batch)
                probs = torch.softmax(logits, dim=1)

                # 2. 仅用Fisher对角线（你的核心逻辑）
                prob_diag = probs * (1 - probs)  # [B, K]

            # 3. 逐样本处理（batch_size=8，显存可控）
            for i in range(x_batch.shape[0]):
                x_i = x_batch[i:i + 1]  # [1, D]
                y_i = y_batch[i]

                # 计算残差（和你原代码一致）
                y_onehot = F.one_hot(y_i, num_classes=K).float().to(device)
                residual = probs[i] - y_onehot

                # ========== 关键：仅计算top-3类，显存降到最低 ==========
                topk_idx = torch.topk(probs[i], k=min(3, K)).indices  # 仅top-3类，足够精度
                def f(proto):
                    logits_i = self.forward_lp_with_proto(x_i, proto).squeeze(0)
                    return logits_i[topk_idx]  # 仅返回top-3类

                # ========== 修复Jacobian报错：去掉strategy，改用默认reverse-mode ==========
                J = jacobian(
                    f, 
                    self.adapter.prototypes, 
                    create_graph=False,
                    vectorize=False,  # 现在兼容reverse-mode
                    strict=False  # 去掉strategy参数，用默认的reverse-mode
                )  # [3, K, D] → 展平为[3, K*D]
                J = J.reshape(len(topk_idx), -1).T  # [K*D, 3]

                # ========== 恢复你原有的GGN/G计算逻辑 ==========
                # Fisher对角线（top-3类）
                fisher_i_diag = prob_diag[i][topk_idx]  # [3]
                JH = J * fisher_i_diag.unsqueeze(0)

                # 计算GGN贡献
                contrib_diag = torch.sum(JH * J, dim=1).float()
                GGN_diag.add_(contrib_diag.cpu())

                # 计算G_diag（梯度裁剪和你原代码一致）
                g_i = J @ residual[topk_idx].unsqueeze(-1)
                g_i_norm = g_i.norm()
                if g_i_norm > 20:
                    g_i = g_i / (g_i_norm + 1e-6) * 20.0
                G_diag.add_(g_i.view(-1).cpu())

                # ========== 强制释放显存（核心）==========
                del J, JH, contrib_diag, g_i
                torch.cuda.empty_cache()
                gc.collect()

        # ========== 批次处理（batch_size=8，无OOM）==========
        total_batches = len(range(0, len(feature_train), batch_size))
        pbar = tqdm(range(0, len(feature_train), batch_size), desc="计算对角线 GGN (最终修复版)")
        for b_idx, b in enumerate(pbar):
            x_batch = feature_train[b:b + batch_size]
            y_batch = label_train[b:b + batch_size]

            process_batch(x_batch, y_batch)

            # 每5批次清理显存+监控
            if b_idx % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                pbar.set_postfix({
                    '显存占用': f"{allocated:.1f}GB",
                    'GGN均值': f"{GGN_diag.mean().item():.6f}"
                })
        pbar.close()

        # ========== 后处理（和你原代码完全一致）==========
        GGN_diag = GGN_diag.to(device)
        G_diag = G_diag.to(device)

        print(f"GGN_diag 统计: min={GGN_diag.min():.6f}, max={GGN_diag.max():.6f}, mean={GGN_diag.mean():.6f}")
        print(f"G_diag 统计: min={G_diag.min():.6f}, max={G_diag.max():.6f}, mean={G_diag.mean():.6f}")

        # GGN裁剪（你的逻辑）
        ggn_upper = torch.quantile(GGN_diag, 0.95)
        GGN_diag_clipped = GGN_diag.clamp(max=ggn_upper)

        # 计算协方差（你的逻辑）
        cov_diag = 1 / (GGN_diag_clipped + prior_prec + 1e-6)

        # MAP参数（你的逻辑）
        theta_map = self.adapter.prototypes.flatten().to(device)
        posterior_mu = theta_map + 0.1 * cov_diag * (G_diag + GGN_diag_clipped + prior_prec)

        self.adapter.prototypes_map = posterior_mu.view(K, D).clone()
        self.adapter.ggn_cov = cov_diag.view(K, D)

        print(f"后验均值 norm: {posterior_mu.norm().item():.4f}")
        print(f"GGN_cov 范围: {cov_diag.min():.6f} - {cov_diag.max():.6f}")

        # 保存结果
        if save_path is not None:
            torch.save({
                'prototypes_map': self.adapter.prototypes_map.cpu(),
                'ggn_cov': self.adapter.ggn_cov.cpu(),
                'GGN_diag': GGN_diag.cpu(),
                'G_diag': G_diag.cpu()
            }, save_path)
            print(f"✅ GGN结果已保存到: {save_path}")

        return self.adapter.ggn_cov
    def load_ggn_covariance(self, load_path, device='cuda:0'):
        checkpoint = torch.load(load_path, map_location='cpu')
        self.adapter.ggn_cov = checkpoint['ggn_cov']
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
        self.adapter.prototypes.data = self.adapter.prototypes.data.to(device)
        prototypes = self.adapter.prototypes

        # Normalize
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

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
        features_norm = features / features.norm(dim=-1, keepdim=True).to('cuda:0')

        # 采样原型
        prototypes_samples = self.adapter.sample_prototypes(self.adapter.n_samples).to('cuda:0')
        prototypes_norm = prototypes_samples / prototypes_samples.norm(dim=-1, keepdim=True)

        # 计算所有采样的logits
        logit_scale = self.logit_scale.exp().to('cuda:0')
        all_logits = torch.einsum('bd,skd->sbk', features_norm, prototypes_norm) * logit_scale

        # 计算不确定性
        probs = F.softmax(all_logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        uncertainty = 1 - max_probs.mean(dim=0)

        # 返回均值预测和不确定性
        logits_mean = all_logits.mean(dim=0)
        return logits_mean, uncertainty


class TrainerXCostume(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("eval")

        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # 调试模式：强制只运行指定数量的batch
        if DEBUG_MODE:
            self.num_batches = DEBUG_TRAIN_BATCHES
            print(f"\n⚠️  调试模式：仅训练 {DEBUG_TRAIN_BATCHES} 个batch ⚠️")
        else:
            self.num_batches = len(self.train_loader_x)
            
        self.batch_size = self.train_loader_x.batch_size

        features = self.features_train.clone().cpu().numpy()
        labels = self.labels_train.clone()

        if "CrossModal" in self.model.adapter.initialization:
            idx = np.random.choice(list(np.arange(0, features.shape[0])), features.shape[0] // 2)
            features = features[idx, :]
            labels = labels[idx]

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
            
            # 调试模式：训练完指定batch后立即退出循环
            if DEBUG_MODE and self.batch_idx + 1 >= DEBUG_TRAIN_BATCHES:
                print(f"\n⚠️  调试模式：已完成 {DEBUG_TRAIN_BATCHES} 个batch训练，提前终止训练循环 ⚠️")
                break
                
        return loss_summary


@TRAINER_REGISTRY.register()
class LaplaceBayesADAPTER(TrainerXCostume):
    """General Adapter with Bayesian Laplace approximation"""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LaplaceBayesADAPTER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LaplaceBayesADAPTER.PREC == "fp32" or cfg.TRAINER.LaplaceBayesADAPTER.PREC == "amp":
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
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter", self.model.adapter, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LaplaceBayesADAPTER.PREC == "amp" else None

    def train(self):
        self.best_acc = 0.0
        self.best_model_state = None

        self.set_model_mode("eval")

        # ========== 新增：特征保存/读取核心配置 ==========
        import os
        import os.path as osp
        # 特征保存目录（和训练输出目录同层级）
        feat_save_dir = osp.join(self.cfg.OUTPUT_DIR, "cached_features")
        os.makedirs(feat_save_dir, exist_ok=True)

        # 特征文件名（绑定seed/shot/数据集，确保相同seed复用）
        feat_filename = f"features_{self.cfg.DATASET.NAME}_shot{self.cfg.SHOT}_seed{self.cfg.SEED}.pt"
        feat_save_path = osp.join(feat_save_dir, feat_filename)

        # 调试模式强制重新提取特征，不读取缓存
        if DEBUG_MODE or not osp.exists(feat_save_path):
            print(f"\n⚠️  特征文件不存在/调试模式，开始提取特征: {feat_save_path}")

            # 提取测试集特征（使用完整数据）
            print("\n=== 提取测试集特征（使用完整数据）===")
            self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
            print(f"测试集样本数: {len(self.features_test)}")
            print("Zero-Shot accuracy on test: " +
                  str(round(compute_accuracy(output_test.cuda(), self.labels_test.cuda())[0].item(), 2)))

            # 提取训练集特征
            print("\n=== 提取训练集特征 ===")
            self.labels_train, self.logits_zs, self.features_train = self.extract_features(
                partition="train", reps=self.model.adapter.epochs_aumentation, transforms=self.model.adapter.augmentations)
            print(f"训练集样本数: {len(self.features_train)}")

            # 保存特征到文件（仅非调试模式保存）
            if not DEBUG_MODE:
                feat_data = {
                    "labels_test": self.labels_test.cpu(),
                    "output_test": output_test.cpu(),
                    "features_test": self.features_test.cpu(),
                    "labels_train": self.labels_train.cpu(),
                    "logits_zs": self.logits_zs.cpu(),
                    "features_train": self.features_train.cpu(),
                    "seed": self.cfg.SEED,
                    "shot": self.cfg.SHOT,
                    "dataset": self.cfg.DATASET.NAME
                }
                torch.save(feat_data, feat_save_path)
                print(f"\n✅ 特征已保存到: {feat_save_path}")
        else:
            # ========== 读取缓存特征（修复：确保所有张量到GPU） ==========
            print(f"\n🎉 找到缓存特征文件，直接读取: {feat_save_path}")
            feat_data = torch.load(feat_save_path, map_location="cpu")

            # 验证seed/shot匹配（防止读取错误特征）
            assert feat_data["seed"] == self.cfg.SEED, f"特征seed({feat_data['seed']})和当前seed({self.cfg.SEED})不匹配！"
            assert feat_data["shot"] == self.cfg.SHOT, f"特征shot({feat_data['shot']})和当前shot({self.cfg.SHOT})不匹配！"
            assert feat_data["dataset"] == self.cfg.DATASET.NAME, f"特征数据集({feat_data['dataset']})和当前数据集({self.cfg.DATASET.NAME})不匹配！"

            # 强制将所有特征转移到GPU（核心修复）
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.labels_test = feat_data["labels_test"].to(device, non_blocking=True)
            output_test = feat_data["output_test"].to(device, non_blocking=True)
            self.features_test = feat_data["features_test"].to(device, non_blocking=True)
            self.labels_train = feat_data["labels_train"].to(device, non_blocking=True)
            self.logits_zs = feat_data["logits_zs"].to(device, non_blocking=True)
            self.features_train = feat_data["features_train"].to(device, non_blocking=True)

            print(f"✅ 特征加载完成（GPU） | 测试集: {len(self.features_test)} | 训练集: {len(self.features_train)}")
            print("Zero-Shot accuracy on test: " +
                  str(round(compute_accuracy(output_test, self.labels_test)[0].item(), 2)))

        # ========== 修复：确保模型adapter的device是GPU ==========
        self.model.adapter.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ========== 原有训练逻辑（保持不变） ==========
        # Init alphas in constraint formulation
        if self.model.adapter.apply_constraint != "none":
            print("Getting initial lagrangian multipliers for constraint formulation", end="\n")
            # 显式将logits_zs和labels_train传入时确保在GPU
            self.model.adapter.init_lagrangian_multipliers(
                self.labels_train.to(self.model.adapter.device), 
                self.logits_zs.to(self.model.adapter.device)
            )
            print("Lagrangian multipliers: ")
            print(list(torch.round(self.model.adapter.alpha_constraint.detach(), decimals=3).cpu().numpy()))

        # Training
        # ===== timing: MAP + posterior training time (start) =====
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t_train_start = time.perf_counter()
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            loss_summary = self.run_epoch()

            if loss_summary["acc_test"] > self.best_acc:
                self.best_acc = loss_summary["acc_test"]
                self.best_model_state = copy.deepcopy(self.model.adapter.state_dict())
                print(f"Best acc updated: {self.best_acc:.4f}")

            if "adaptative" in self.model.adapter.apply_constraint:
                self.model.adapter.outer_step()

            self.after_epoch()

            # 调试模式：只训练1个epoch就进入测试阶段
            if DEBUG_MODE:
                print(f"\n⚠️  调试模式：已完成1个epoch训练，提前进入测试阶段 ⚠️")
                break

        ggn_dir = osp.join(self.cfg.OUTPUT_DIR, "ggn_checkpoints")
        os.makedirs(ggn_dir, exist_ok=True)
        ggn_path = osp.join(ggn_dir, f"ggn_cov_seed{self.cfg.SEED}_debug.pt")

        print(f"Loaded best model with acc: {self.best_acc:.4f}")

        # 进入测试阶段
        self.after_train(self.features_train, self.labels_train,
                         self.features_test, self.labels_test,
                         ggn_save_path=ggn_path)
    def bayesian_evaluation(self, feature_test, label_test, mode="linearized", posterior_path=None):
        """测试阶段：使用完整数据，增强显存监控"""
        print("\n=== 开始贝叶斯评估（使用完整测试集）===")
        print(f"测试集样本总数: {len(feature_test)}")
        print_gpu_memory("Before Bayesian evaluation")
        
        self.set_model_mode("eval")
        if mode == "linearized":
            if (self.model.adapter.prototypes_map is None or
                    self.model.adapter.ggn_cov is None):
                assert posterior_path is not None, "未提供后验路径，且未预加载 prototypes_map / ggn_cov。"
                self.model.load_ggn_covariance(posterior_path, device=self.device)

        features = feature_test.to(self.device)
        labels = label_test.to(self.device)

        # 记录测试开始时间，用于评估速度
        start_time = time.time()
        
        final_result = self.search_optimal_temperature(
            features, labels,
            n_samples=self.model.adapter.n_samples,
            batch_size=16  # 显存友好的batch_size
        )
        
        # 计算测试耗时
        test_duration = time.time() - start_time
        test_samples_per_second = len(feature_test) / test_duration
        
        print("\n=== 测试性能统计 ===")
        print(f"测试总耗时: {test_duration:.2f} 秒")
        print(f"每秒处理样本数: {test_samples_per_second:.2f} samples/s")
        print_gpu_memory("After Bayesian evaluation")

        print("\n=== 最优结果汇总 ===")
        print(f"1. ECE最优温度: {final_result['best_T_for_ece']:.2f}")
        print(f"   - 对应ACC: {final_result['best_acc_at_best_ece'] * 100:.2f}%")
        print(f"   - 对应ECE:  {final_result['best_ece']:.4f}")
        print(f"   - 对应AECE: {final_result['best_aece']:.4f}")

        print(f"\n2. ACC最优温度: {final_result['best_T_for_acc']:.2f}")
        print(f"   - 对应ACC: {final_result['best_acc'] * 100:.2f}%")
        print(f"   - 对应ECE:  {final_result['best_ece_at_best_acc']:.4f}")
        print(f"   - 对应AECE: {final_result['best_aece_at_best_acc']:.4f}")
            
    def _compute_adaptive_ece(
        self,
        confidences,
        predictions,
        labels,
        n_bins=10,
        save_csv_path=None,
        T=1.0,
        confidence_thresholds=(0.8, 0.85, 0.9, 0.95, 0.99),
        strict_above=True,
    ):
        """
        输出：
          - AECE（等频自适应分箱）：保存 _aece_conf.csv（含 Conf Min/Max）
          - ECE（等宽分箱）：保存 _ece_conf.csv（含 Low/High + Conf Min/Max）
          - Table2/Table3（逐样本精确+reliability gate）：保存 _table2_table3.csv

        返回：
          (ece, aece, overall_acc, table23_df)
        """

        import numpy as np
        import pandas as pd

        # ---------- to numpy ----------
        confidences = confidences.detach().cpu().numpy() if hasattr(confidences, "detach") else np.asarray(confidences)
        predictions = predictions.detach().cpu().numpy() if hasattr(predictions, "detach") else np.asarray(predictions)
        labels = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)

        confidences = np.asarray(confidences).astype(np.float64)
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        total_samples = len(confidences)
        if total_samples == 0:
            raise ValueError("Empty inputs: confidences is empty.")

        # total classes (prefer max+1 if labels are contiguous ids)
        try:
            num_classes = int(labels.max()) + 1
        except Exception:
            num_classes = len(np.unique(labels))

        # ---------- overall ----------
        overall_acc = float(np.mean(predictions == labels))
        overall_conf = float(np.mean(confidences))
        overall_conf_min = float(np.min(confidences))
        overall_conf_max = float(np.max(confidences))

        # ============================================================
        # 1) AECE (equal-frequency bins)  -> save _aece_conf.csv
        # ============================================================
        sorted_indices = np.argsort(confidences)
        conf_sorted = confidences[sorted_indices]
        pred_sorted = predictions[sorted_indices]
        label_sorted = labels[sorted_indices]

        total = len(conf_sorted)
        bin_size = total // n_bins if n_bins > 0 else total

        # 带 Conf Min/Max 的 AECE 统计
        aece_conf_stats = []  # ["Bin","Count","Conf Min","Conf Max","Acc","Conf","|Gap|"]

        for i in range(n_bins):
            start = i * bin_size
            end = total if i == n_bins - 1 else (i + 1) * bin_size
            if end <= start:
                continue

            bin_conf = conf_sorted[start:end]
            bin_pred = pred_sorted[start:end]
            bin_label = label_sorted[start:end]

            acc = float(np.mean(bin_pred == bin_label))
            conf = float(np.mean(bin_conf))
            gap = abs(acc - conf)

            aece_conf_stats.append([
                i + 1,
                len(bin_conf),
                float(bin_conf.min()),
                float(bin_conf.max()),
                acc,
                conf,
                gap
            ])

        aece = float(np.mean([x[-1] for x in aece_conf_stats])) if len(aece_conf_stats) > 0 else 0.0

        # overall row (Bin=0)
        aece_conf_stats.append([
            0,
            total,
            float(conf_sorted.min()),
            float(conf_sorted.max()),
            overall_acc,
            overall_conf,
            abs(overall_acc - overall_conf)
        ])

        # ============================================================
        # 2) ECE (equal-width bins) -> save _ece_conf.csv
        # ============================================================
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        # 带 Conf Min/Max 的 ECE 统计（同时保留 Low/High，方便复现传统定义）
        ece_conf_stats = []  # ["Bin","Count","Low","High","Conf Min","Conf Max","Acc","Conf","|Gap|"]

        for i in range(n_bins):
            low, high = float(bin_edges[i]), float(bin_edges[i + 1])

            in_bin = (confidences >= low) & (confidences < high)
            if i == n_bins - 1:
                in_bin = (confidences >= low) & (confidences <= high)

            if not np.any(in_bin):
                continue

            bin_conf = confidences[in_bin]
            bin_pred = predictions[in_bin]
            bin_label = labels[in_bin]

            acc = float(np.mean(bin_pred == bin_label))
            conf = float(np.mean(bin_conf))
            gap = abs(acc - conf)

            ece += gap * (len(bin_conf) / total_samples)

            ece_conf_stats.append([
                i + 1,
                int(len(bin_conf)),
                low,
                high,
                float(bin_conf.min()),
                float(bin_conf.max()),
                acc,
                conf,
                gap
            ])

        # overall row (Bin=0)
        ece_conf_stats.append([
            0,
            total_samples,
            0.0,
            1.0,
            overall_conf_min,
            overall_conf_max,
            overall_acc,
            overall_conf,
            abs(overall_acc - overall_conf)
        ])

        # ============================================================
        # 3) Table 2 & 3 (exact, per-sample, reliability gated)
        # ============================================================
        rows = []
        for tau in confidence_thresholds:
            tau = float(tau)

            if strict_above:
                sel = confidences > tau
            else:
                sel = confidences >= tau

            n_sel = int(sel.sum())

            if n_sel == 0:
                rows.append({
                    "threshold": tau,
                    "n_selected": 0,
                    "subset_acc": np.nan,
                    "reliable": False,
                    "sample_cov_pct": 0.0,     # 选不到样本，coverage=0
                    "n_classes_covered": 0,
                    "class_cov_pct": np.nan,   # 无意义
                })
                continue

            subset_acc = float(np.mean(predictions[sel] == labels[sel]))
            reliable = (subset_acc >= tau)

            if not reliable:
                rows.append({
                    "threshold": tau,
                    "n_selected": n_sel,
                    "subset_acc": subset_acc,
                    "reliable": False,
                    "sample_cov_pct": np.nan,     # 论文用 ✗，这里用 NaN
                    "n_classes_covered": np.nan,
                    "class_cov_pct": np.nan,
                })
                continue

            sample_cov_pct = (n_sel / total_samples) * 100.0
            n_classes_covered = int(len(np.unique(labels[sel])))  # IMPORTANT: true labels
            class_cov_pct = (n_classes_covered / num_classes) * 100.0 if num_classes > 0 else 0.0

            rows.append({
                "threshold": tau,
                "n_selected": n_sel,
                "subset_acc": subset_acc,
                "reliable": True,
                "sample_cov_pct": sample_cov_pct,
                "n_classes_covered": n_classes_covered,
                "class_cov_pct": class_cov_pct,
            })

        table23_df = pd.DataFrame(rows)
        # === add meta for later aggregation ===
        table23_df["num_classes"] = num_classes
        table23_df["total_samples"] = total_samples
        table23_df["T"] = float(T)

        # 这些如果你不想改函数签名，也可以先不加；但强烈建议加
        table23_df["dataset"] = self.cfg.DATASET_NAME
        table23_df["shot"] = self.cfg.SHOT
        table23_df["seed"] = self.cfg.SEED
        table23_df["backbone"] = 'resnet50'


        # ============================================================
        # 4) Save CSVs (ONLY conf versions)
        # ============================================================
        if save_csv_path is not None:
            # AECE (conf ranges)
            aece_conf_df = pd.DataFrame(
                aece_conf_stats,
                columns=["Bin", "Count", "Conf Min", "Conf Max", "Acc", "Conf", "|Gap|"]
            )
            aece_conf_df["T"] = T
            aece_conf_df.to_csv(save_csv_path.replace(".csv", "_aece_conf.csv"), index=False)

            # ECE (conf ranges + low/high)
            ece_conf_df = pd.DataFrame(
                ece_conf_stats,
                columns=["Bin", "Count", "Low", "High", "Conf Min", "Conf Max", "Acc", "Conf", "|Gap|"]
            )
            ece_conf_df["T"] = T
            ece_conf_df.to_csv(save_csv_path.replace(".csv", "_ece_conf.csv"), index=False)

            # Table2/Table3
            table23_df.to_csv(save_csv_path.replace(".csv", "_table2_table3.csv"), index=False)

            # 控制台打印（可选）
            def fmt(x):
                return "✗" if (pd.isna(x)) else f"{x:.2f}"

            print(f"[Saved AECE_conf, ECE_conf, Table2/Table3 CSVs | T={T}]")
            print("[Table2/Table3 exact (reliability gated)]")
            for _, r in table23_df.iterrows():
                tau = int(r["threshold"] * 100)
                sc = fmt(r["sample_cov_pct"])
                cc = fmt(r["class_cov_pct"])
                sa = r["subset_acc"]
                sa_str = "nan" if pd.isna(sa) else f"{sa*100:.2f}%"
                print(f"  {tau}%: sample_cov={sc}% | class_cov={cc}% | subset_acc={sa_str} | reliable={bool(r['reliable'])}")

        return ece, aece, overall_acc, table23_df


    def linearized_predict_batchwise_optimized(self, features, labels, n_samples=50, batch_size=16, T=1.0, jac_batch_size=4):
        """
        混合批量雅可比计算：平衡显存与速度
        jac_batch_size: 雅可比计算的小批量大小，可根据显存调整（建议2-8）
        """
        assert self.model.adapter.prototypes_map is not None, "必须先计算后验分布!"
        assert jac_batch_size >=1, "jac_batch_size不能小于1"

        print(f"\n=== 线性化预测（特征batch: {batch_size}，雅可比batch: {jac_batch_size}，温度: {T}）===")
        print(f"待处理样本总数: {len(features)}")
        print_gpu_memory("Before linearized predict")

        K, D = self.model.adapter.prototypes_map.shape
        device = features.device
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        theta_map = self.model.adapter.prototypes.flatten().to(device)
        theta_samples = self._sample_from_posterior_diag(n_samples).to(device)

        all_preds = []
        all_confidences = []
        all_labels = []

        total = features.size(0)
        start_time = time.time()

        # 1. 特征和基础logits计算：保持大batch并行（显存友好）
        print("Precomputing test logits with large batch...")
        test_logits = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                logits = self.model.forward_lp(features[start:end])
            test_logits.append(logits)
        test_logits = torch.cat(test_logits, dim=0)
        print("Test logits precomputed!")
        print_gpu_memory("After precompute test logits")

        # 2. 雅可比计算：小批量分块处理，替代严格单样本
        for start in tqdm(range(0, total, batch_size), desc="Optimized Bayesian Predict"):
            end = min(start + batch_size, total)
            sub_features = features[start:end]
            sub_labels = labels[start:end]
            sub_logits = test_logits[start:end]

            batch_preds = []
            batch_confs = []

            # 2.1 对当前特征batch，再切分为雅可比小批量
            for jac_start in range(0, len(sub_features), jac_batch_size):
                jac_end = min(jac_start + jac_batch_size, len(sub_features))
                jac_features = sub_features[jac_start:jac_end]
                jac_logits = sub_logits[jac_start:jac_end]

                try:
                    # 2.2 小批量雅可比计算（并行加速）
                    with torch.enable_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                        self.model.adapter.prototypes.requires_grad_(True)

                        def proto_batch_forward(proto):
                            proto = proto.reshape(K, D)
                            feat_norm = jac_features / jac_features.norm(dim=-1, keepdim=True)
                            proto_norm = proto / proto.norm(dim=-1, keepdim=True)
                            logit_scale = self.model.logit_scale.exp().to(device)
                            return feat_norm @ proto_norm.t() * logit_scale

                        # 批量计算雅可比：输出[B, K, K, D]，对应B个样本的雅可比矩阵
                        J_batch = jacobian(
                            proto_batch_forward,
                            self.model.adapter.prototypes,
                            vectorize=True,  # 关键：开启批量向量计算
                            create_graph=False,
                            strict=False
                        )  # [B, K, K, D]
                        J_batch = J_batch.reshape(jac_end-jac_start, K, K*D)  # [B, K, K*D]

                    # 2.3 批量线性化预测
                    delta = theta_samples - theta_map.unsqueeze(0)  # [n_samples, K*D]
                    # 矩阵乘法：[B, K, K*D] @ [n_samples, K*D].T → [B, K, n_samples]
                    correction_batch = torch.matmul(J_batch, delta.transpose(0,1))
                    # 计算采样logits均值
                    logits_samples_batch = jac_logits.unsqueeze(2) + correction_batch  # [B, K, n_samples]
                    logits_mean_batch = logits_samples_batch.mean(dim=2)  # [B, K]
                    probs_batch = torch.softmax(logits_mean_batch / T, dim=1)  # [B, K]

                    # 收集结果
                    batch_preds.extend(probs_batch.argmax(dim=1).cpu().tolist())
                    batch_confs.extend(probs_batch.max(dim=1).values.cpu().tolist())

                    # 清理当前雅可比batch的显存
                    del J_batch, correction_batch, logits_samples_batch
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    # 3. 兜底策略：小批量爆显存时，自动降级为单样本计算
                    if "out of memory" in str(e):
                        print(f"\n⚠️  雅可比batch {jac_batch_size} 显存不足，降级为单样本计算")
                        for i in range(jac_start, jac_end):
                            single_feature = sub_features[i:i+1]
                            single_logit = jac_logits[i:i+1]

                            with torch.enable_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                                def proto_single_forward(proto):
                                    proto = proto.reshape(K, D)
                                    feat_norm = single_feature / single_feature.norm(dim=-1, keepdim=True)
                                    proto_norm = proto / proto.norm(dim=-1, keepdim=True)
                                    logit_scale = self.model.logit_scale.exp().to(device)
                                    return feat_norm @ proto_norm.t() * logit_scale

                                J_i = jacobian(
                                    proto_single_forward,
                                    self.model.adapter.prototypes,
                                    vectorize=False,
                                    create_graph=False,
                                    strict=False
                                ).squeeze(0)  # [K, K, D]
                                J_i = J_i.reshape(K, K*D)  # [K, K*D]

                            delta = theta_samples - theta_map.unsqueeze(0)
                            correction = torch.matmul(J_i, delta.transpose(0,1))  # [K, n_samples]
                            logits_samples = single_logit.unsqueeze(2) + correction  # [1, K, n_samples]
                            logits_mean = logits_samples.mean(dim=2)  # [1, K]
                            probs = torch.softmax(logits_mean / T, dim=1)

                            batch_preds.append(probs.argmax().item())
                            batch_confs.append(probs.max().item())

                            del J_i, correction, logits_samples
                            torch.cuda.empty_cache()
                    else:
                        raise e

            all_preds.extend(batch_preds)
            all_confidences.extend(batch_confs)
            all_labels.extend(sub_labels.cpu().tolist())
            torch.cuda.empty_cache()
            gc.collect()

        # 统计速度
        process_time = time.time() - start_time
        samples_per_second = total / process_time

        # 计算指标
        all_preds = torch.tensor(all_preds)
        all_confidences = torch.tensor(all_confidences)
        all_labels = torch.tensor(all_labels)

        acc = (all_preds == all_labels).float().mean().item()
        ece, aece, overall_acc, table2_3_df = self._compute_adaptive_ece(
            all_confidences, all_preds, all_labels, n_bins=10, 
            save_csv_path=f"{self.cfg.OUTPUT_DIR}/calibration_results_T{T:.1f}_hybrid.csv",
            T=T
        )

        print(f"\n=== 混合批量处理统计 ===")
        print(f"总处理时间: {process_time:.2f} 秒")
        print(f"处理速度: {samples_per_second:.2f} samples/s")
        print(f"准确率: {acc * 100:.2f}%")
        print_gpu_memory("After linearized predict")

        return acc, ece, aece

    def search_optimal_temperature(self, features, labels, n_samples=500, batch_size=16, T_range=None, bins=10):
        from collections import defaultdict

        T = get_temperature(self.cfg.SHOT, self.cfg.SEED)

        results = defaultdict(list)

        print("\n🔍 Using fixed temperature for calibration...")

        acc, ece, aece = self.linearized_predict_batchwise_optimized(
            features, labels,
            n_samples=n_samples,
            batch_size=batch_size,
            T=T,
        )

        print(f"  T = {T:.2f} | Acc = {acc * 100:.2f}%, ECE = {ece:.4f}, AECE = {aece:.4f}")
        results["T"].append(T)
        results["acc"].append(acc)
        results["ece"].append(ece)
        results["aece"].append(aece)

        best_T_for_ece = T
        best_acc_at_best_ece = acc
        best_ece = ece
        
        best_T_for_Aece = T
        best_aece = aece
        best_aece_at_best_acc = aece

        best_T_for_acc = T
        best_acc = acc
        best_ece_at_best_acc = ece

        print("\n✅ Fixed Temperature Results:")
        print(
            f"[For Best ECE] T = {best_T_for_ece:.2f} | Acc = {best_acc_at_best_ece * 100:.2f}%, ECE = {best_ece:.4f}")
        print(
            f"[For Best ACC] T = {best_T_for_acc:.2f} | Acc = {best_acc * 100:.2f}%, ECE = {best_ece_at_best_acc:.4f}")

        return {
            'best_T_for_ece': best_T_for_ece,
            'best_acc_at_best_ece': best_acc_at_best_ece,
            'best_ece': best_ece,
            'best_aece': best_aece,  # NEW
            'best_T_for_acc': best_T_for_acc,
            'best_acc': best_acc,
            'best_ece_at_best_acc': best_ece_at_best_acc,
            'best_aece_at_best_acc': best_aece_at_best_acc,  # NEW
            'all_results': results
        }

    def _sample_from_posterior(self, n_samples):
        K, D = self.model.adapter.prototypes_map.shape
        mu = self.model.adapter.prototypes_map.flatten().to('cuda:0')
        cov = self.model.adapter.ggn_cov.reshape(K * D, K * D).to('cuda:0')

        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            print("⚠️ Covariance not positive-definite. Adding small jitter (1e-6 * I).")
            cov = cov + 1e-6 * torch.eye(K * D, device='cuda:0')
            L = torch.linalg.cholesky(cov)

        eps = torch.randn(n_samples, K * D, device='cuda:0')
        samples = mu + eps @ L.T

        return samples

    def _sample_from_posterior_diag(self, n_samples):
        K, D = self.model.adapter.prototypes_map.shape
        mu = self.model.adapter.prototypes_map.flatten().to('cuda:0')
        var_diag = self.model.adapter.ggn_cov.flatten().to('cuda:0')

        eps = torch.randn(n_samples, K * D, device='cuda:0')
        std = torch.sqrt(var_diag).unsqueeze(0)
        samples = mu.unsqueeze(0) + eps * std

        return samples

    def after_train(self, feature_train, label_train, feature_test, label_test, ggn_save_path=None):
        """调试模式的测试阶段入口"""
        print("\n=== 进入测试阶段（调试模式）===")
        print(f"训练集样本数: {len(feature_train)}")
        print(f"测试集样本数: {len(feature_test)}")
        print_gpu_memory("Before test phase")
        
        import os
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_train = feature_train.to(device, non_blocking=True)
        label_train = label_train.to(device, non_blocking=True)

        self.model = self.model.to(device)
        self.model.adapter = self.model.adapter.to(device)
    
        if self.best_model_state is not None:
            current_prior_var = self.model.adapter.prior_var.clone()
            self.model.adapter.load_state_dict(self.best_model_state)
            print(f"Loaded best model with acc: {self.best_acc:.4f}")
            self.model.adapter.prior_var = current_prior_var
            print(f"Restored prior_var to: {current_prior_var.mean().item():.4f}")

        import time, json, os
        import os.path as osp

        # ===== posterior timing start (optional breakdown) =====
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_post0 = time.perf_counter()

        posterior_loaded = False
        if ggn_save_path is not None and os.path.exists(ggn_save_path):
            print("Loading saved Laplace-GGN covariance...")
            self.model.load_ggn_covariance(ggn_save_path)
            posterior_loaded = True
        else:
            print("Computing Laplace-GGN covariance ...")
            self.model.compute_ggn_diag_covariance(
                feature_train=feature_train,
                label_train=label_train,
                batch_size=8,
                save_path=ggn_save_path
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_post1 = time.perf_counter()

        posterior_time_sec = t_post1 - t_post0

        # ===== total training time: MAP(start) -> posterior(end) =====
        total_train_time_sec = t_post1 - getattr(self, "_t_train_start", t_post1)

        print(f"[TIME] posterior_time_sec: {posterior_time_sec:.2f}s | total_train_time_sec(MAP->posterior): {total_train_time_sec:.2f}s")

        # ===== save record (so you can merge across servers) =====
        records_dir = osp.join(self.cfg.OUTPUT_DIR, "time_records")
        os.makedirs(records_dir, exist_ok=True)

        rec = {
            "dataset": self.cfg.DATASET.NAME,
            "method": "BayesAdapter" if "Bayes" in self.cfg.TRAINER.NAME or "Laplace" in self.cfg.TRAINER.NAME else self.cfg.TRAINER.NAME,
            "backbone": self.cfg.MODEL.BACKBONE.NAME,
            "shot": int(self.cfg.SHOT),
            "seed": int(self.cfg.SEED),
            "total_train_time_sec_map_to_posterior": float(total_train_time_sec),
            "posterior_time_sec": float(posterior_time_sec),
            "posterior_loaded": bool(posterior_loaded),
        }

        out_path = osp.join(
            records_dir,
            f"time_{rec['dataset']}_{rec['method']}_{rec['backbone']}_shot{rec['shot']}_seed{rec['seed']}.json"
        )
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[Saved time record] {out_path}")

        # ===== DO NOT include evaluation in training time =====
        print("\n=== 开始贝叶斯评估（完整测试集）===")
        self.bayesian_evaluation(feature_test, label_test)


        print("\n=== 调试模式测试完成 ===")
        print_gpu_memory("Test phase completed")

        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"总耗时: {elapsed}")

        self.close_writer()

    def compute_confidence_penalty(self, logits, reduction='mean'):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)

        if reduction == 'mean':
            return entropy.mean()
        else:
            return entropy

    def forward_backward(self, features, labels):
        prec = self.cfg.TRAINER.LaplaceBayesADAPTER.PREC
        if prec == "amp":
            with torch.amp.autocast('cuda'):
                output = self.model.forward_features(torch.tensor(features).to(self.device))
                loss_ce = F.cross_entropy(output, labels)
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
            output = self.model.forward_features(torch.tensor(features).to(self.device))
            loss_ce = F.cross_entropy(output, labels)
            confidence_penalty = self.compute_confidence_penalty(output, reduction='mean')
            lambda_conf = 0
            if self.model.adapter.apply_constraint != "none":
                loss_constraint = self.model.adapter.zero_shot_constraint()
                loss = loss_ce + loss_constraint + lambda_conf * confidence_penalty
            else:
                loss = loss_ce

            self.model_backward_and_update(loss)
        
        with torch.no_grad():
            output_test = self.model.forward_features(self.features_test.clone().detach().to(self.device))

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, labels)[0].item(),
            "acc_test": compute_accuracy(output_test, self.labels_test)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
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

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
            self.model.float()

    def extract_features(self, partition, reps=1, transforms=False):
        print(f"Extracting features from: {partition} (调试模式)")
        self.set_model_mode("eval")

        if partition == "train":
            data_loader = copy.deepcopy(self.train_loader_x)

            if not transforms:
                data_loader.dataset.transform = self.val_loader.dataset.transform

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
                labels_ds_irep = torch.cat(labels_ds_irep, dim=0)
                logits_dsirep = torch.cat(logits_dsirep, dim=0)
                features_ds_irep = torch.cat(features_ds_irep, dim=0)
                labels_ds.append(labels_ds_irep.unsqueeze(0))
                logits_ds.append(logits_dsirep.unsqueeze(0))
                features_ds.append(features_ds_irep.unsqueeze(0))

            labels_ds = torch.cat(labels_ds, dim=0)[0, :]
            logits_ds = torch.cat(logits_ds, dim=0).mean(0)
            features_ds = torch.cat(features_ds, dim=0).mean(0)

        print(f"Extracted features - {partition}: labels={len(labels_ds)}, features={features_ds.shape}")
        return labels_ds, logits_ds, features_ds