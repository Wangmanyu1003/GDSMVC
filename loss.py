import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F
import math

class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f,temperature_l ,class_num, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.class_num = class_num
        self.mask = self.mask_correlated_samples(batch_size).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity = nn.CosineSimilarity(dim=2)

    def uniform_uncertainty_loss(self, q):
        eps = 1e-12
        q = q.to(self.device).float()
        q = q.clamp_min(eps)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        entropy = -(q * q.log()).sum(dim=1)
        loss = (math.log(self.class_num) - entropy).mean() * self.class_num
        return loss


    def distribution_alignment_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        让两种分布(形状 N x C)尽可能接近：逐样本计算 Jensen–Shannon Divergence 并平均。
        - 无需超参数
        - 自动行归一化，数值稳定
        - 返回标量，越小表示 q 与 p 越接近
        """
        eps = 1e-12
        q = q.to(self.device).float().clamp_min(eps)
        p = p.to(self.device).float().clamp_min(eps)

        # 行归一化，保证是合法分布
        q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)

        m = 0.5 * (q + p)

        # KL(q||m) 和 KL(p||m)
        kl_qm = (q * (q.log() - m.log())).sum(dim=1)  # (N,)
        kl_pm = (p * (p.log() - m.log())).sum(dim=1)  # (N,)

        jsd = 0.5 * (kl_qm + kl_pm)  # (N,)
        # 为了量级更友好，可按类别数做自适应缩放（不是超参，只依赖 C）
        return (jsd.mean() * self.class_num)

    def Structure_guided_Contrastive_Loss(self, h_i, h_j,size):
        #S_1 = S.repeat(2, 2)
        all_one = torch.ones(size*2, size*2).to('cuda')
        #S_2 = all_one - S_1
        S_2 = all_one
        N = 2 * size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_l
        sim1 = torch.multiply(sim, S_2)
        sim_i_j = torch.diag(sim, size)
        sim_j_i = torch.diag(sim, -size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        # ---------- 统一把 q_i, q_j 变成 float 概率矩阵 (N, C) ----------
        eps = 1e-12

        # p_i = q_i.sum(0).view(-1)
        # p_i /= p_i.sum()
        # ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        # p_j = q_j.sum(0).view(-1)
        # p_j /= p_j.sum()
        # ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        # entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_f
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask
