import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal timestep embedding (as before)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# Cosine schedule for betas (as before)
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

def reparameterize_timesteps(alphas_bar, diffusion_steps):
    """
    Given the training cumulative product alphas_bar (shape: (T,)),
    compute a tensor of timestep indices (length: diffusion_steps)
    that are approximately uniformly spaced in the alpha domain.
    """
    T = alphas_bar.size(0)
    new_alphas = torch.linspace(alphas_bar[0].item(), alphas_bar[-1].item(), diffusion_steps)
    new_t = []
    for a in new_alphas:
        diff = torch.abs(alphas_bar - a)
        idx = torch.argmin(diff)
        new_t.append(idx.item())
    return torch.tensor(new_t, device=alphas_bar.device)
