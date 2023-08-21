import torch.optim as optim
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
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
        return result, mean, std

    def sampling(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        res = mean + eps * std
        return res
    
def save_model(model, optimizer, path_to_save='/home/tigran/Desktop/cv_interview/autoencoders/vae/vae50_model.pth'):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path_to_save)


def vae_loss(recon_x, x, mean, log_var):
    RECON = F.mse_loss(recon_x.view(-1, (784)), x.view(-1, (784)), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RECON + KLD, RECON, KLD

def criterion(x, x_hat, mean, log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence



tr = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

trn_ds = MNIST('./data', download=True, transform=tr, train=True)
val_ds = MNIST('./data', train=False, download=True, transform=tr)

train_loader = DataLoader(trn_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True)

epochs=50

model = VAE(8)
vae_loss = criterion
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    print(epoch + 1)
    total_loss = 0
    for i, (inputs, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_hat, mean, log_var = model(inputs)
        loss = vae_loss(x_hat, inputs, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(trn_ds):.4f}")

save_model(model, optimizer)
print('model.saved')

# Generate new images from random samples in the latent space
# with torch.no_grad():
#     model.eval()
#     n = 10
#     digit_size = 28
#     figure = torch.zeros((digit_size * n, digit_size * n))

#     for i, yi in enumerate(torch.linspace(-3, 3, n)):
#         for j, xi in enumerate(torch.linspace(-3, 3, n)):
#             z_sample = torch.tensor([[xi, yi]])
#             x_decoded = model.decode(z_sample)
#             digit = x_decoded.view(digit_size, digit_size)
#             figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

#     save_image(figure, "generated_images.png", normalize=True)

# import cv2
# import numpy as np
# with torch.no_grad():
#     for i, (inputs, _) in enumerate(val_dl):
#         x_hat, mean, log_var = model(inputs)
#         print(x_hat.shape)
#         img = inputs.squeeze()
#         pred = np.array(x_hat.squeeze())
#         print(img.shape)
#         cv2.imshow('pred', pred)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         break


            

