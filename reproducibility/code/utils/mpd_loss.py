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
        self.gamma_history = []  

    def update_epoch(self, epoch):
        """Implement update_epoch."""
        self.current_epoch = epoch
        current_gamma = self.get_dynamic_gamma()
        self.gamma_history.append(current_gamma)

        # W&B logging
        if wandb.run is not None:
            wandb.log({"MPD Gamma": current_gamma}, step=epoch)

    def get_dynamic_gamma(self):
        """Implement get_dynamic_gamma."""
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

    
