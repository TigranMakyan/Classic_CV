import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Set random seed for reproducibility
torch.manual_seed(42)

# Define VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        latent_params = self.encoder(x.view(x.size(0), -1))
        mu = latent_params[:, :latent_dim]
        logvar = latent_params[:, latent_dim:]

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

# Set hyperparameters
latent_dim = 20
epochs = 20
batch_size = 128
learning_rate = 1e-3

# Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = MNIST(root='./data', train=True, download=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create VAE instance and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define loss function
def vae_loss(reconstruction, x, mu, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstruction, x.view(x.size(0), -1), reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (x, _) in enumerate(train_dataloader):
        x = x.to(device)

        # Forward pass
        reconstruction, mu, logvar = model(x)

        # Compute loss
        loss = vae_loss(reconstruction, x, mu, logvar)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_dataset):.4f}")

# Generate new samples from the VAE
model.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    generated_samples = model.decoder(z).view(-1, 1, 28, 28)

# You can use the generated_samples as desired (e.g., visualize or save them)
