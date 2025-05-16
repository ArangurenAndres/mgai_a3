import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os

# -----------------------------
# Config Loader
# -----------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# -----------------------------
# Diffusion noise schedule
# -----------------------------
class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

# -----------------------------
# Positional Embedding for time t
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# -----------------------------
# UNet with per-layer time embedding projections
# -----------------------------
class MarioUNet(nn.Module):
    def __init__(self, in_channels=10, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        self.time_mlp1 = nn.Sequential(nn.Linear(time_emb_dim, base_channels), nn.ReLU())
        self.time_mlp2 = nn.Sequential(nn.Linear(time_emb_dim, base_channels * 2), nn.ReLU())
        self.time_mlp_mid = nn.Sequential(nn.Linear(time_emb_dim, base_channels * 2), nn.ReLU())
        self.time_mlp_dec = nn.Sequential(nn.Linear(time_emb_dim, base_channels), nn.ReLU())

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Middle
        self.middle = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)  # [B, time_emb_dim]
        emb1 = self.time_mlp1(t_emb).unsqueeze(-1).unsqueeze(-1)      # [B, C, 1, 1]
        emb2 = self.time_mlp2(t_emb).unsqueeze(-1).unsqueeze(-1)
        emb_mid = self.time_mlp_mid(t_emb).unsqueeze(-1).unsqueeze(-1)
        emb_dec = self.time_mlp_dec(t_emb).unsqueeze(-1).unsqueeze(-1)

        h1 = F.relu(self.enc1(x) + emb1)
        h2 = F.relu(self.enc2(self.pool(h1)) + emb2)
        mid = F.relu(self.middle(h2) + emb_mid)

        d1 = F.relu(self.dec1(self.up(mid)) + emb_dec)
        out = self.out(d1)
        return out

# -----------------------------
# Full Diffusion Wrapper
# -----------------------------
class DiffusionModel(nn.Module):
    def __init__(self, unet, schedule):
        super().__init__()
        self.unet = unet
        self.schedule = schedule

    def forward(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self.schedule.q_sample(x_0, t, noise)
        predicted_noise = self.unet(x_t, t)
        return F.mse_loss(predicted_noise, noise)

# -----------------------------
# Builder from config
# -----------------------------
def build_diffusion_model_from_config(config_path="config.yaml", device="cpu"):
    config = load_config(config_path)
    model_cfg = config["model"]

    unet = MarioUNet(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        time_emb_dim=model_cfg["time_emb_dim"]
    ).to(device)

    schedule = DiffusionSchedule(
        timesteps=model_cfg["diffusion_steps"],
        beta_start=model_cfg["beta_start"],
        beta_end=model_cfg["beta_end"],
        device=device
    )

    model = DiffusionModel(unet, schedule).to(device)
    return model, schedule

# -----------------------------
# Run to print model config + structure
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, "..", "config.yaml"))

    config = load_config(config_path)
    model_cfg = config["model"]

    print("✅ Model Configuration:")
    for k, v in model_cfg.items():
        print(f"  {k}: {v}")

    model, _ = build_diffusion_model_from_config(config_path, device)

    print("\n✅ Model Structure:")
    print(model.unet)
