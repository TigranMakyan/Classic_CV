from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms as T
from model import AutoEncoder
from torchsummary import summary

img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

trn_ds = MNIST('./data', download=True, transform=img_transform, train=True)
val_ds = MNIST('./data', train=False, download=True, transform=img_transform)

BATCH_SIZE=8

trn_dl = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

model = AutoEncoder()
summary(model, (1, 28, 28))


