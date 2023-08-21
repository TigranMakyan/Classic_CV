import numpy as np
from sklearn.metrics import precision_score, confusion_matrix

def iou(pred_mask, mask):
    intersection = np.logical_and(pred_mask, mask)
    union = np.logical_or(pred_mask, mask)
    return np.sum(intersection) / np.sum(union)

def dice(pred_mask, mask):
    intersection = np.logical_and(pred_mask, mask)
    dice = (2 * np.sum(intersection)) / np.sum(pred_mask) + np.sum(mask)
    return dice

def pixel_acc(pred_mask, mask):
    size = pred_mask.shape[-1] * pred_mask.shape[-2]
    new = np.where(pred_mask==mask, 1, 0)
    acc = np.sum(new) / size
    return acc

def calculate_precision(pred_mask, gt_mask):
    true_positive = np.logical_and(pred_mask, gt_mask)
    predicted_positive = np.sum(pred_mask)
    precision = np.sum(true_positive) / predicted_positive
    return precision

def calculate_recall(pred_mask, gt_mask):
    true_positive = np.logical_and(pred_mask, gt_mask)
    predicted_positive = np.sum(gt_mask)
    recall = np.sum(true_positive) / predicted_positive
    return recall

def calculate_f_measure(pred_mask, gt_mask):
    true_positive = np.logical_and(pred_mask, gt_mask)
    precision = np.sum(true_positive) / np.sum(pred_mask)
    recall = np.sum(true_positive) / np.sum(gt_mask)
    
    f_measure = (2 * precision * recall) / (precision + recall)
    return f_measure

def calculate_mean_precision(pred_masks, gt_masks):
    bag = []
    for i in range(12):
        _mask = np.where(pred_masks == i, i, 0)
        _gt_mask = np.where(gt_masks == i, i, 0)
        bag.append(calculate_precision(_mask, _gt_mask))

    return bag

