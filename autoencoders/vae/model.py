import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(32*7*7, latent_dim)
        self.fc_var = nn.Linear(32*7*7, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32*7*7),
            nn.ReLU(True),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.fc_mean(x)
        std = self.fc_var(x)
        return mean, std

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, std = self.encode(x)
        latent = self.sampling(mean, std)
        result = self.decode(latent)
        return result

    def sampling(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)