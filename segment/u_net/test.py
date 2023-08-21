import torch
from torch_snippets import *

from data import SegData
from model import UNet

trn_ds = SegData('train')
val_ds = SegData('test')
trn_dl = DataLoader(trn_ds, batch_size=4, collate_fn=trn_ds.collate_fn, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, collate_fn=val_ds.collate_fn)

model = UNet()
checkpoint = torch.load('/home/tigran/Desktop/cv_interview/segment/u_net/unet.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

im, mask = next(iter(val_dl))
_mask = model(im)
_, _mask = torch.max(_mask, dim=1)
# subplots([im[0].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
# ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],
# nc=3, titles=['Original image','Original mask','Predicted mask'])

print(im.shape)
print(_mask.shape)

from metrics import iou_score, pixel_accuracy, dice_score

iou = iou_score(pred_mask=_mask, true_mask=mask)   #===0.8
acc = pixel_accuracy(_mask, mask)
dice = dice_score(_mask, mask)
print(f'IoU score: {iou}')
print(f'Pixel accuracy score: {acc}')
print(f'DICE score: {dice}')
