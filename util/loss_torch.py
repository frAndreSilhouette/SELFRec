import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd


def bpr_loss(user_emb, pos_item_emb, neg_item_emb, itts=None, loss_func=0):
    if loss_func == 0:
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    
    else:
        print('[ERROR] bpr_loss: loss_func is not defined.')
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

class QuantileBPRLoss(nn.Module):
    def __init__(self, n_items, loss_func):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kappa_fixed = 1.0
        self.kappa = nn.Parameter(torch.ones(n_items, device=self.device))
        self.theta = nn.Parameter(torch.zeros(n_items, device=self.device))
        self.loss_func = loss_func
        print(f'[DEBUG] self.loss_func: {self.loss_func}')
    
    def calculate_cdf(self, item_ids, days, scales, shapes, all_items: bool = True) -> torch.Tensor:
        """
        根据给定的 scales 和 shapes（与 item_ids 对齐），计算 Weibull CDF
        """
        # 自动转 tensor
        days = days.to(self.device).float()

        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        else:
            scales = scales.to(self.device).float()

        if not isinstance(shapes, torch.Tensor):
            shapes = torch.tensor(shapes, dtype=torch.float32, device=self.device)
        else:
            shapes = shapes.to(self.device).float()

        dist = torch.distributions.Weibull(scales, shapes)
        cdf_vals = dist.cdf(days)

        if not all_items:
            mask = shapes < 1
            return torch.where(mask, torch.full_like(cdf_vals, -1.0), cdf_vals)
        else:
            return cdf_vals


    def calculate_inverse_cdf(self, item_ids, cdf_values, scales, shapes, all_items: bool = True) -> torch.Tensor:
        """
        根据给定的 scales 和 shapes（与 item_ids 对齐），计算 Weibull 逆CDF
        """
        # 自动转 tensor
        cdf_values = cdf_values.to(self.device).float()

        if not isinstance(scales, torch.Tensor):
            scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        else:
            scales = scales.to(self.device).float()

        if not isinstance(shapes, torch.Tensor):
            shapes = torch.tensor(shapes, dtype=torch.float32, device=self.device)
        else:
            shapes = shapes.to(self.device).float()

        # 避免数值问题
        cdf_values = torch.clamp(cdf_values, min=1e-6, max=1 - 1e-6)

        dist = torch.distributions.Weibull(scales, shapes)
        icdf_vals = dist.icdf(cdf_values)

        if not all_items:
            mask = shapes < 1
            return torch.where(mask, torch.full_like(icdf_vals, -1.0), icdf_vals)
        else:
            return icdf_vals

    def forward(self, user_emb, pos_item_emb, neg_item_emb, pos_idx, neg_idx, itts, scales, shapes):
        
        itts = torch.tensor(itts, dtype=torch.float32, device=self.device)
        pos_cdfs = self.calculate_cdf(pos_idx, itts, scales, shapes)
        qs = torch.sigmoid(self.theta)
        pos_qs = qs[pos_idx]
        quantile_thresholds = self.calculate_inverse_cdf(pos_idx, pos_qs, scales, shapes)
        pos_kappa = self.kappa[pos_idx]

        # print("pos_kappa device:", pos_kappa.device)
        # print("itts device:", itts.device)
        # print("quantile_thresholds device:", quantile_thresholds.device)

        if self.loss_func == 0:
            # 原版
            pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        elif self.loss_func == 1:
            # kappa固定，分位数之差
            pos_weights = torch.sigmoid(self.kappa_fixed * (itts - quantile_thresholds))
            pos_scores = torch.sum(user_emb * pos_item_emb * pos_weights.unsqueeze(1), dim=1)
        elif self.loss_func == 2:
            # kappa可学习，分位数之差
            pos_weights = torch.sigmoid(pos_kappa * (itts - quantile_thresholds))
            pos_scores = torch.sum(user_emb * pos_item_emb * pos_weights.unsqueeze(1), dim=1)
        elif self.loss_func == 3:
            # kappa可学习，分位数之比
            pos_weights = torch.sigmoid(pos_kappa * (itts / (quantile_thresholds + 1e-6)))
            pos_scores = torch.sum(user_emb * pos_item_emb * pos_weights.unsqueeze(1), dim=1)
        elif self.loss_func == 4:
            # kappa可学习，cdf之差
            pos_weights = torch.sigmoid(pos_kappa * (pos_cdfs - pos_qs))
            pos_scores = torch.sum(user_emb * pos_item_emb * pos_weights.unsqueeze(1), dim=1)
        elif self.loss_func == 5:
            # kappa可学习，cdf之比
            pos_weights = torch.sigmoid(pos_kappa * (pos_cdfs / (pos_qs + 1e-6)))
            pos_scores = torch.sum(user_emb * pos_item_emb * pos_weights.unsqueeze(1), dim=1)
        else:
            print('[ERROR] bpr_loss: loss_func is not defined.')

        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_scores - neg_scores))

        return torch.mean(loss)

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = ((user_emb-pos_item_emb)**2).sum(dim=1)
    neg_score = ((user_emb-neg_item_emb)**2).sum(dim=1)
    loss = F.relu(pos_score-neg_score+0.5)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


#this version is from recbole
def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    if sim == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)

    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return F.cross_entropy(logits, labels)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

