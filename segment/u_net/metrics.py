import torch

def iou_score(pred_mask, true_mask, smooth=1e-6):
    intersection = (pred_mask & true_mask).sum(dim=(1, 2))
    union = (pred_mask | true_mask).sum(dim=(1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
    
def fw_iou_score(pred_mask, true_mask, num_classes, smooth=1e-6):
    intersection = (pred_mask & true_mask).sum(dim=(1, 2))
    union = (pred_mask | true_mask).sum(dim=(1, 2))
    fw_iou = torch.zeros(num_classes, device=pred_mask.device)
    
    for class_idx in range(num_classes):
        class_mask = true_mask == class_idx
        class_intersection = (pred_mask == class_idx) & class_mask
        class_union = (pred_mask == class_idx) | class_mask
        class_iou = (class_intersection.sum(dim=(1, 2)) + smooth) / (class_union.sum(dim=(1, 2)) + smooth)
        class_weight = class_mask.sum(dim=(1, 2))
        fw_iou[class_idx] = (class_iou * class_weight).sum() / class_weight.sum()

    return fw_iou.mean().item()


def dice_score(pred_mask, true_mask, smooth=1e-6):
    intersection = (pred_mask & true_mask).sum(dim=(1, 2))
    pred_sum = pred_mask.sum(dim=(1, 2))
    true_sum = true_mask.sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / (pred_sum + true_sum + smooth)
    return dice.mean().item()
    
def pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum().item()
    total = pred_mask.numel()
    accuracy = correct / total
    return accuracy
