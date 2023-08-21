import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as T

m = nn.ConvTranspose2d(3, 3, kernel_size=(2, 2), stride=2, padding=0)
img = cv2.imread('image.jpg')
print(img.shape)

tr = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.225, 0.224])
])
img = tr(img)
print(img.shape)
# print(inp.shape)
out = m(img)
print(out.shape)
asd = out.permute(1, 2, 0)
asd = asd.detach().numpy()
# cv2.imshow('image', asd)
# cv2.waitKey(0)
# cv2.destroyAllWindows()