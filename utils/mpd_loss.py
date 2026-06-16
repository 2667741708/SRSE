# # utils/mpd_loss.py (最终融合版)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MPDLoss(nn.Module):
    # """
    # 一个融合了 SimSiam 和 Barlow Twins 思想的统一自监督损失模块。
    
    # 1. SimSiam 部分 (预测): 通过负余弦相似度，拉近一个视图的预测 (p1) 和另一个视图的特征 (z2)。
    # 2. Barlow Twins 部分 (差异化): 通过冗余降低正则项，推开不同特征通道之间的相关性。
    # """
    # def __init__(self, feature_dim: int, lmbda_barlow: float = 5e-3, gamma_barlow: float = 0.1):
    #     """
    #     初始化 MPD 损失模块。
        
    #     Args:
    #         feature_dim (int): SimSiam 模型投影头 (Projector) 输出的特征维度。
    #         lmbda_barlow (float): Barlow Twins 正则项内部的 lambda 参数。
    #         gamma_barlow (float): 平衡 SimSiam 损失和 Barlow Twins 正则项的权重。
    #     """
    #     super().__init__()
    #     self.lmbda_barlow = lmbda_barlow
    #     self.gamma_barlow = gamma_barlow
        
    #     # 创建一个可复用的 BatchNorm 层，用于 Barlow Twins 的计算
    #     self.bn = nn.BatchNorm1d(feature_dim, affine=False)

    # def simsiam_loss_fn(self, p, z):
    #     # 将 SimSiam 的损失计算封装在内部
    #     return -F.cosine_similarity(p, z, dim=-1).mean()

    # def redundancy_reduction_loss(self, z):
    #     # 将 Barlow Twins 的冗余降低损失计算封装在内部
    #     batch_size, feature_dim = z.shape
        
    #     # 对特征进行中心化和标准化
    #     z_norm = self.bn(z)
        
    #     # 计算自相关矩阵
    #     corr_matrix = z_norm.T @ z_norm
    #     corr_matrix.div_(batch_size)
        
    #     # 计算非对角线元素的惩罚
    #     def off_diagonal(x):
    #         n, m = x.shape
    #         assert n == m
    #         return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    #     off_diag_loss = off_diagonal(corr_matrix).pow_(2).sum()
        
    #     return self.lmbda_barlow * off_diag_loss

    # def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    #     """
    #     计算总的 MPD 损失。
        
    #     Args:
    #         p1, p2: SimSiam 模型预测头 (Predictor) 的输出。
    #         z1, z2: SimSiam 模型投影头 (Projector) 的输出 (已 detach)。
            
    #     Returns:
    #         torch.Tensor: 计算出的总损失。
    #     """
    #     # --- 1. 计算 SimSiam 部分 (预测损失) ---
    #     # 注意：输入的 z1, z2 应该已经在外部 detach 过了
    #     loss_simsiam = self.simsiam_loss_fn(p1, z2) / 2 + self.simsiam_loss_fn(p2, z1) / 2
        
    #     # --- 2. 计算 Barlow Twins 部分 (差异化损失) ---
    #     # 对两个视图的特征 z1 和 z2 分别计算冗余，然后取平均
    #     loss_barlow_reg = (self.redundancy_reduction_loss(z1) + self.redundancy_reduction_loss(z2)) / 2
        
    #     # --- 3. 组合最终损失 ---
    #     total_loss = loss_simsiam + self.gamma_barlow * loss_barlow_reg
        
    #     return total_loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb


class MPDLoss(nn.Module):
    def __init__(
        self,
        feature_dim,
        gamma_barlow=0.1,
        lambda_barlow=5e-3,
        total_epochs=500,
        schedule_gamma=True,
    ):
        super().__init__()
        self.gamma_barlow_max = gamma_barlow
        self.lambda_barlow = lambda_barlow
        self.total_epochs = total_epochs
        self.schedule_gamma = schedule_gamma
        self.current_epoch = 0
        self.gamma_history = []  # 存储 gamma 随时间变化

    def update_epoch(self, epoch):
        """每个 epoch 调用一次，更新 gamma 并记录"""
        self.current_epoch = epoch
        current_gamma = self.get_dynamic_gamma()
        self.gamma_history.append(current_gamma)
        print(f"[Epoch {epoch:03d}] Dynamic gamma = {current_gamma:.5f}")

        # W&B logging
        if wandb.run is not None:
            wandb.log({"MPD Gamma": current_gamma}, step=epoch)

    def get_dynamic_gamma(self):
        """返回当前 epoch 的 gamma 值"""
        if not self.schedule_gamma:
            return self.gamma_barlow_max
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        gamma = self.gamma_barlow_max * (1 - math.cos(math.pi * progress)) / 2
        return gamma

    def simsiam_loss(self, p, z):
        """SimSiam loss: negative cosine similarity"""
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def barlow_loss(self, z1, z2):
        """Barlow Twins loss"""
        N, D = z1.size()
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
        c = torch.mm(z1_norm.T, z2_norm) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()).pow_(2).sum()
        return on_diag + self.lambda_barlow * off_diag

    def forward(self, p1, p2, z1, z2):
        loss_sim = (self.simsiam_loss(p1, z2) + self.simsiam_loss(p2, z1)) / 2
        loss_barlow = self.barlow_loss(z1, z2)
        gamma = self.get_dynamic_gamma()
        total_loss = loss_sim + gamma * loss_barlow
        return total_loss

    