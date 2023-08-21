import torch
from torch_snippets import *

from data import SegData
from model import UNet

trn_ds = SegData('train')
val_ds = SegData('test')
trn_dl = DataLoader(trn_ds, batch_size=4, collate_fn=trn_ds.collate_fn, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, collate_fn=val_ds.collate_fn)

ce = nn.CrossEntropyLoss()
def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc

def train_batch(model, loss, data, optimizer):
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss_value, acc = loss(_masks, ce_masks)
    loss_value.backward()
    optimizer.step()
    return loss_value.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data, loss):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss_value, acc = loss(_masks, masks)
    return loss_value.item(), acc.item()

model = UNet()
print(model)
criterion = UnetLoss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 20

log = Report(n_epochs)
for ex in range(n_epochs):
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss, acc = train_batch(model=model, loss=criterion, optimizer=optimizer, data=data)
        log.record(ex+(bx + 1)/ N, trn_loss=loss, trn_acc=acc, end='\r')

    N = len(val_dl)
    for bx, data in enumerate(val_dl):
        loss, acc = validate_batch(model=model, data=data, loss=criterion)
        log.record(ex+(bx+1)/N, val_loss=loss, val_acc=acc, end='\r')
    
    log.report_avgs(ex+1)

torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, '/home/tigran/Desktop/cv_interview/segment/u_net/best_unet.pth')

log.plot_epochs(['trn_loss', 'val_loss'])


