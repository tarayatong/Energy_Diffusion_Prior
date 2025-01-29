import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
import math

def SoftIoULoss(pred, target):
    # Old One
    pred = torch.sigmoid(pred)  # B 1 256 256
    smooth = 1
    #miou/iou
    intersection = pred * target    # B 1 256 256
    loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)
    #niou
    # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
    #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
    #         - intersection.sum(axis=(1, 2, 3)) + smooth)
    loss = 1 - loss.mean()
    # loss = (1 - loss).mean()
    return loss

def BCEDiceLoss(pred, target):
    target = ((target-F.avg_pool2d(target, kernel_size=5, stride=1, padding=2))>0).float()
    smooth = 1
    pred = pred.mean(1, True)
    # target = F.interpolate(target, size=pred.shape[-2:], mode='nearest')
    # if target.shape != pred.shape:
    #     target = F.max_pool2d(target, kernel_size=(target.shape[-1]//pred.shape[-1])+1, stride=(target.shape[-1]//pred.shape[-1]), padding=math.ceil((target.shape[-1]//pred.shape[-1]-1)/2))
    bce = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1-dice + bce.mean())/2

def wsoftmiou_loss(pred, target, edge):
    weit  = 1+5*torch.abs(F.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)-edge)
    pred = torch.sigmoid(pred)  # B 1 256 256
    smooth = 1
    #miou/iou
    inter = (pred * target *weit).sum()    # B 1 256 256
    union = ((pred+target) *weit).sum()
    loss = (inter + smooth) / (union -inter + smooth)
    loss = 1 - loss.mean()  # 什么维度
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


