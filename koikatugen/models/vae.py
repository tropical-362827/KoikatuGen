import torch
import torch.nn as nn
import torch.nn.functional as F

from koikatugen.models.base import BaseModel


class VAE(BaseModel):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mean(x), self.fc_logvar(x)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc_decode(z))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim)
        with torch.no_grad():
            return self.decode(z)
