import torch
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam

from preds.gradients import Jacobians

import torch

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: Tensor of shape [batch_size, num_classes] on any device
    labels: Tensor of shape [batch_size] on same device as logits
    returns: (accuracy: Tensor with shape [], correct: int, total: int)
    """
    # 预测类别索引
    predicted = torch.argmax(logits, dim=1)

    # 计算预测正确的数量
    correct = (predicted == labels).sum()
    total = labels.size(0)

    # 准确率为 float 类型 Tensor
    accuracy = correct.float() / total

    return accuracy


def GGN(model, likelihood, data, target=None, ret_f=False):
    Js, f = Jacobians(model, data)
    if target is not None:
        rs = likelihood.residual(target, f)
    Hess = likelihood.Hessian(f)
    m, p = Js.shape[:2]
    if len(Js.shape) == 2:
        k = 1
        Hess = Hess.reshape(m, k, k)
        if target is not None:
            rs = rs.reshape(m, k)
        Js = Js.reshape(m, p, k)
    if target is not None:
        if ret_f:
            return Js, Hess, rs, f
        return Js, Hess, rs
    else:
        return Js, Hess, f


def expand_prior_mu(prior_mu, P, device):
    if type(prior_mu) is float:
        return torch.ones(P, device=device) * prior_mu
    elif type(prior_mu) is torch.Tensor:
        return prior_mu
    else:
        raise ValueError('Invalid shape for prior mean')


def expand_prior_prec(prior_prec, P, device):
    if type(prior_prec) is float or prior_prec.ndim == 0:
        prec_diag = torch.ones(P, device=device) * prior_prec
        return torch.diag(prec_diag), torch.diag(1 / prec_diag)
    elif prior_prec.ndim == 1:
        return torch.diag(prior_prec), torch.diag(1 / prior_prec)
    elif prior_prec.ndim == 2:
        return prior_prec, torch.inverse(prior_prec)
    else:
        raise ValueError('Invalid shape for prior precision')


class LaplaceGGN(Adam):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), prior_prec=1.0, prior_mu=0.0,
                 eps=1e-8, amsgrad=False, **kwargs):
        if 'beta1' in kwargs and 'beta2' in kwargs:
            betas = (kwargs['beta1'], kwargs['beta2'])

        weight_decay = 0
        super().__init__(model.parameters(), lr, betas, eps, weight_decay, amsgrad)

        p = parameters_to_vector(model.parameters())
        P = len(p)
        device = p.device
        self.defaults['device'] = device

        # 设置先验精度矩阵
        P_0, S_0 = expand_prior_prec(prior_prec, P, device)
        self.state['prior_prec'] = P_0
        self.state['Sigma_0'] = S_0

        # 设置先验均值
        if isinstance(prior_mu, torch.Tensor):
            assert prior_mu.shape[0] == P, f"Expected prior_mu of shape ({P},), got {prior_mu.shape}"
            self.state['prior_mu'] = prior_mu.to(device)
        else:
            self.state['prior_mu'] = torch.full((P,), float(prior_mu), device=device)

        self.state['mu'] = None
        self.state['precision'] = None
        self.state['Sigma_chol'] = None

    def step(self, closure):
        # compute gradients on network using our standard closures
        log_lik,f,y_train = closure()
        #计算ACC
        acc_train = compute_accuracy(f, y_train)
        print("acc_train:",acc_train.item())
        params = parameters_to_vector(self.param_groups[0]['params'])
        prior_mu = self.state['prior_mu']
        prior_prec = self.state['prior_prec']

        diff = params - prior_mu
        weight_loss = 0.5 * diff @ prior_prec @ diff
        print('负对数似然：',-log_lik.item())
        loss = -log_lik + 0.0001 * weight_loss
        print('总损失：', loss.item())
        loss.backward()
        super(LaplaceGGN, self).step()
        return loss.item()

    def post_process(self, model, likelihood, train_loader):
        device = self.defaults['device']
        parameters = self.param_groups[0]['params'] # 需要更新的参数（均值向量）
        theta_star = parameters_to_vector(parameters).to(device) # 参数的 MAP 值就是均值向量了
        prior_prec = self.state['prior_prec'] # 先验的协方差精度
        prior_mu = self.state['prior_mu'] # 先验的均值
        P = len(theta_star)
        JLJ = torch.zeros(P, P, device=device)
        G = torch.zeros(P, device=device)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            Js, Hess, rs = GGN(model, likelihood, data, target)
            '''
             JΛJ 就是 Hessian 矩阵的近似，去掉了参数的二阶导部分
             在batch上做累加，最终得到 GGN Hessian 的全精度近似
            '''
            JLJ += torch.einsum('mpk,mkl,mql->pq', Js, Hess, Js)
            G += torch.einsum('mpk,mk->p', Js, rs)
        # compute posterior covariance and precision
        # 公式（12）
        self.state['precision'] = JLJ + prior_prec
        Chol = torch.cholesky(self.state['precision'])
        self.state['Sigma'] = torch.cholesky_inverse(Chol, upper=False)
        self.state['Sigma_chol'] = torch.cholesky(self.state['Sigma'], upper=False)
        # compute posterior mean according to BLR/Exact Laplace
        b = G + JLJ @ theta_star + prior_prec @ prior_mu
        self.state['mu'] = torch.cholesky_solve(b.reshape(-1, 1), Chol, upper=False).flatten()
        return self


def get_diagonal_ggn(optimizer):
    diag_prec = torch.diag(optimizer.state['precision'])
    Sigma_diag = 1 / diag_prec
    Sigma_chol = torch.diag(torch.sqrt(Sigma_diag))
    Sigma = torch.diag(Sigma_diag)
    return Sigma, Sigma_chol


class PrototypeLaplaceGGN(LaplaceGGN):
    """专门用于优化prototypes参数的LaplaceGGN变体"""

    def __init__(self, prototypes, lr=0.1, betas=(0.9, 0.999), prior_prec=1.0,
                 prior_mu=0.0, eps=1e-8, amsgrad=False, **kwargs):
        # 将prototypes包装成一个参数组
        param_group = {'params': [prototypes]}
        super(LaplaceGGN, self).__init__([param_group], lr, betas, eps, 0, amsgrad)

        # 初始化Laplace特定的状态
        p = prototypes.data.flatten()
        P = len(p)
        device = p.device
        self.defaults['device'] = device

        # 扩展先验参数
        if isinstance(prior_prec, (float, int)) or (isinstance(prior_prec, torch.Tensor) and prior_prec.ndim == 0):
            prec_diag = torch.ones(P, device=device) * prior_prec
            P_0 = torch.diag(prec_diag)
            S_0 = torch.diag(1 / prec_diag)
        elif prior_prec.ndim == 1:
            P_0 = torch.diag(prior_prec)
            S_0 = torch.diag(1 / prior_prec)
        elif prior_prec.ndim == 2:
            P_0 = prior_prec
            S_0 = torch.inverse(prior_prec)
        else:
            raise ValueError('Invalid shape for prior precision')

        self.state['prior_prec'] = P_0
        self.state['Sigma_0'] = S_0
        self.state['prior_mu'] = prior_mu if isinstance(prior_mu, torch.Tensor) else torch.ones(P,
                                                                                                device=device) * prior_mu
        self.state['mu'] = None
        self.state['precision'] = None
        self.state['Sigma'] = None
        self.state['Sigma_chol'] = None

    def post_process(self, model, likelihood, train_loader):
        """重写后处理方法，仅处理prototypes参数"""
        device = self.defaults['device']
        prototypes = self.param_groups[0]['params'][0]
        theta_star = prototypes.data.flatten().to(device)

        prior_prec = self.state['prior_prec']
        prior_mu = self.state['prior_mu']
        P = len(theta_star)

        JLJ = torch.zeros(P, P, device=device)
        G = torch.zeros(P, device=device)

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            Js, Hess, rs = self._compute_jacobians(model, likelihood, data, target)

            # 仅处理prototypes相关的Jacobians
            Js_prototype = Js[:, :P]  # 假设prototypes参数在前P个位置
            JLJ += torch.einsum('mpk,mkl,mql->pq', Js_prototype, Hess, Js_prototype)
            G += torch.einsum('mpk,mk->p', Js_prototype, rs)

        # 计算后验精度和协方差
        self.state['precision'] = JLJ + prior_prec
        Chol = torch.cholesky(self.state['precision'])
        self.state['Sigma'] = torch.cholesky_inverse(Chol, upper=False)
        self.state['Sigma_chol'] = torch.cholesky(self.state['Sigma'], upper=False)

        # 计算后验均值
        b = G + JLJ @ theta_star + prior_prec @ prior_mu
        self.state['mu'] = torch.cholesky_solve(b.reshape(-1, 1), Chol, upper=False).flatten()
        return self

    def _compute_jacobians(self, model, likelihood, data, target):
        """计算Jacobians、Hessian和残差，仅针对当前batch"""
        Js, f = Jacobians(model, data)
        rs = likelihood.residual(target, f)
        Hess = likelihood.Hessian(f)

        # 调整形状
        m, p = Js.shape[:2]
        if len(Js.shape) == 2:
            k = 1
            Hess = Hess.reshape(m, k, k)
            rs = rs.reshape(m, k)
            Js = Js.reshape(m, p, k)

        return Js, Hess, rs

