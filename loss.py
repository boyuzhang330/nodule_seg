#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

smooth = 1.
epsilon = 1e-6


def sad_loss(pred, target, encoder_flag=True):
    """
    AD: atention distillation loss
    : param pred: input prediction
    : param target: input target
    : param encoder_flag: boolean, True=encoder-side AD, False=decoder-side AD
    """
    target = target.detach()
    if (target.size(-1) == pred.size(-1)) and (target.size(-2) == pred.size(-2)):
        # target and prediction have the same spatial resolution
        pass
    else:
        if encoder_flag == True:
            # target is smaller than prediction
            # use consecutive layers with scale factor = 2
            target = F.interpolate(target, scale_factor=2, mode='trilinear')
        else:
            # prediction is smaller than target
            # use consecutive layers with scale factor = 2
            pred = F.interpolate(pred, scale_factor=2, mode='trilinear')

    num_batch = pred.size(0)
    pred = pred.view(num_batch, -1)
    target = target.view(num_batch, -1)
    pred = F.softmax(pred, dim=1)
    target = F.softmax(target, dim=1)
    return F.mse_loss(pred, target)


def dice_loss(pred, target, mask=None):
    """
    DSC loss
    : param pred: input prediction
    : param target: input target
    : param mask: mask of valid regions
    """

    num_category = pred.shape[1]  # containing background and others
    pred = F.softmax(pred, dim=1)  # first perform Softmax on the prediction results; nB, nC, D, H, W
    pred = pred.permute((1, 0, 2, 3, 4)).contiguous()
    loss = None
    if mask is not None:
        mask = mask.view(-1)
        idcs_effective = (mask > 0)
        if torch.sum(idcs_effective) == 0:
            return loss

    for curcategory in range(1, num_category):
        curtarget = (target == curcategory).float()
        curpred = pred[curcategory]
        iflat = curpred.view(-1)
        tflat = curtarget.view(-1)
        if mask is not None:
            iflat = iflat[idcs_effective]
            tflat = tflat[idcs_effective]

        intersection = torch.sum((iflat * tflat))
        curdice = (2. * intersection + smooth) / (torch.sum(iflat) + torch.sum(tflat) + smooth)

        if loss is None:
            loss = curdice
        else:
            loss += curdice
    return 1. - loss


def tversky_loss(pred, target, beta=0.7):
    """
    Tversky loss
    : param pred: input prediction
    : param target: input target
    : param bete: mask of valid regions
    """
    num_category = pred.shape[1]  # containing background and others
    pred = F.softmax(pred, dim=1)  # first perform Softmax on the prediction results; nB, nC, D, H, W
    pred = pred.permute((1, 0, 2, 3, 4)).contiguous()
    loss = None

    for curcategory in range(1, num_category):
        curtarget = (target == curcategory).float()
        curpred = pred[curcategory]
        prob = curpred.view(-1)
        ref = curtarget.view(-1)

        alpha = 1.0 - beta

        tp = (ref * prob).sum()
        fp = ((1 - ref) * prob).sum()
        fn = (ref * (1 - prob)).sum()
        curloss = tp / (tp + alpha * fp + beta * fn)

        if loss is None:
            loss = curloss
        else:
            loss += curloss
    return 1. - loss / float(num_category - 1)


def bin_dice_loss(pred, target):
    """
    DSC loss
    : param pred: input prediction
    : param target: input target
    """
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = torch.sum((iflat * tflat))
    return 1. - ((2. * intersection + smooth) / (torch.sum(iflat) + torch.sum(tflat) + smooth))


def binary_cross_entropy(y_pred, y_true):
    """
    Binary cross entropy loss
    : param y_pred: input prediction
    : param y_true: input target
    """
    y_true = y_true.view(-1).float()
    y_pred = y_pred.view(-1).float()
    return F.binary_cross_entropy(y_pred, y_true)


def bin_focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Focal loss
    : param y_pred: input prediction
    : param y_true: input target
    : param alpha: balancing positive and negative samples, default=0.25
    : param gamma: penalizing wrong predictions, default=2
    """
    # alpha balance weight for unbalanced positive and negative samples
    # clip to prevent NaN's and Inf's
    y_pred_flatten = torch.clamp(y_pred, min=epsilon, max=1. - epsilon)
    y_pred_flatten = y_pred_flatten.view(-1).float()
    y_true_flatten = y_true.detach()
    y_true_flatten = y_true_flatten.view(-1).float()
    loss = 0

    idcs = (y_true_flatten > 0)
    y_true_pos = y_true_flatten[idcs]
    y_pred_pos = y_pred_flatten[idcs]
    y_true_neg = y_true_flatten[~idcs]
    y_pred_neg = y_pred_flatten[~idcs]

    if y_pred_pos.size(0) != 0 and y_true_pos.size(0) != 0:
        # positive samples
        logpt = torch.log(y_pred_pos)
        loss += -1. * torch.mean(torch.pow((1. - y_pred_pos), gamma) * logpt) * alpha

    if y_pred_neg.size(0) != 0 and y_true_neg.size(0) != 0:
        # negative samples
        logpt2 = torch.log(1. - y_pred_neg)
        loss += -1. * torch.mean(torch.pow(y_pred_neg, gamma) * logpt2) * (1. - alpha)

    return loss


def focal_loss(y_pred, y_true, mask=None, alpha=0.25, gamma=2.0):
    """
    Focal loss
    :param y_pred: input prediction
    :param y_true: input target
    :param mask: mask of valid regions
    :param alpha: balancing positive and negative samples, default=0.25
    :param gamma: penalizing wrong predictions, default=2
    """
    # alpha balance weight for unbalanced positive and negative samples
    # clip to prevent NaN's and Inf's
    num_category = y_pred.shape[1]
    alpha_category = [1. - alpha] + [alpha / (num_category - 1.)] * (num_category - 1)
    # first perform Softmax on the prediction results; nB, nC, D, H, W
    y_pred_flatten = F.softmax(y_pred, dim=1)
    y_pred_flatten = torch.clamp(y_pred_flatten, min=epsilon, max=1. - epsilon)
    y_pred_flatten = y_pred_flatten.permute(1, 0, 2, 3, 4).contiguous()
    loss = 0
    y_true_flatten = y_true.detach()
    if mask is not None:
        mask_flatten = mask.view(-1)
        idcs_effective = (mask_flatten > 0)
        if torch.sum(idcs_effective) == 0:
            return loss

    for curcategory in range(0, num_category):
        curtarget = (y_true_flatten == curcategory)
        curpred = y_pred_flatten[curcategory]

        iflat = curpred.view(-1).float()
        tflat = curtarget.view(-1).float()
        if mask is not None:
            iflat = iflat[idcs_effective]
            tflat = tflat[idcs_effective]

        idcs = (tflat > 0)
        tflat_pos = tflat[idcs]
        iflat_pos = iflat[idcs]

        if tflat_pos.size(0) != 0 and iflat_pos.size(0) != 0:
            logpt = torch.log(iflat_pos)
            loss += -1. * torch.mean(torch.pow((1. - iflat_pos), gamma) * logpt) * alpha_category[curcategory]

    return loss


def general_union_loss_yuan(pred, target, dist):
    weight = dist * target + (1 - target)
    # when weight = 1, this loss becomes Root Tversky loss
    smooth = 1.0
    alpha = 0.1  # alpha=0.1 in stage1 and 0.2 in stage2
    beta = 1 - alpha
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target * sigma1 + (1 - target) * sigma2
    intersection = (weight * ((pred + weight_i) ** 0.7) * target).sum()
    intersection2 = (weight * (alpha * pred + beta * target)).sum()
    return 1 - (intersection + smooth) / (intersection2 + smooth)


def general_union_loss(pred, target, dist, mask=None):
    """
    general union loss
    : param pred: input prediction
    : param target: input target
    : param dist: distance of voxel , dist.shape == pred.shape 1~0 ，center and background = 1 ，most distant from center of pred = 0,other = 0~1
    : param mask: mask of valid regions
    """
    num_category = pred.shape[1]  # containing background and others
    pred = F.softmax(pred, dim=1)  # first perform Softmax on the prediction results; nB, nC, D, H, W
    pred = pred.permute((1, 0, 2, 3, 4)).contiguous()
    loss = None
    smooth = 1.0
    alpha = 0.3  # 0.2
    beta = 1 - alpha
    sigma1 = 0.0001
    sigma2 = 0.0001

    if mask is not None:
        mask = mask.view(-1)
        idcs_effective = (mask > 0)
        if torch.sum(idcs_effective) == 0:
            return loss

    for curcategory in range(1, num_category):
        curtarget = (target == curcategory).float()
        # dist = (dist == curcategory).float()
        curpred = pred[curcategory]
        iflat = curpred.view(-1)
        tflat = curtarget.view(-1)
        # dist = dist.view(-1)
        if mask is not None:
            iflat = iflat[idcs_effective]
            tflat = tflat[idcs_effective]
        # weight = dist * tflat + (1 - tflat)
        # weight = 1
        # when weight = 1, this loss becomes Root Tversky loss

        weight_i = tflat * sigma1 + (1 - tflat) * sigma2

        intersection = (((iflat + weight_i) ** 0.7) * tflat).sum()
        # intersection = (dist*((iflat + weight_i) ** 0.7) * tflat).sum()
        intersection2 = ((alpha * iflat + beta * tflat)).sum()
        # intersection2 = (dist*(alpha * iflat + beta * tflat)).sum()

        curloss = (intersection + smooth) / (intersection2 + smooth)
        if loss is None:
            loss = curloss
        else:
            loss += curloss
    return 1. - loss / float(num_category - 1)


def weighted_softmax_cross_entropy_with_logits_ignore_labels(pred, label, pos_weight):
    y = label.clone()
    loss1 = F.cross_entropy(pred, torch.squeeze(y, dim=1).long(), reduction='none')
    y[y == 1] = 1000
    y[y == 0] = 1

    loss2 = torch.mean(torch.mul(loss1,y))
    # loss1 = torch.mean(loss1)
    # print(loss1.requires_grad)
    # print(loss2)
    # print(loss2.requires_grad)

    # print(label)
    # loss2 = F.cross_entropy(pred, torch.squeeze(label,dim=1).long())
    # print(loss2)
    # print(label)
    return loss2


if __name__ == '__main__':
    a = torch.ones([2, 2, 34, 34, 34])
    b = torch.ones([2, 1, 34, 34, 34])
    torch.squeeze(b)
    print(b.shape)
    c = weighted_softmax_cross_entropy_with_logits_ignore_labels(a, b, 50)
    print(c)
