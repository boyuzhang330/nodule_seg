# encoding: utf-8
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
import pickle
import csv
import torch.nn.functional as F
import cv2
import copy
import SimpleITK as sitk
from torch.nn.init import xavier_normal_, kaiming_normal_, constant_, normal_
from loss import bin_dice_loss
from scipy.ndimage.interpolation import zoom




smooth = 1.


def weights_init(net, init_type='normal'):
    """
    :param m: modules of CNNs
    :return: initialized modules
    """

    def init_func(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            if init_type == 'normal':
                normal_(m.weight.data)
            elif init_type == 'xavier':
                xavier_normal_(m.weight.data)
            else:
                kaiming_normal_(m.weight.data)
            if m.bias is not None:
                constant_(m.bias.data, 0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return


def load_pickle(filename='split_dataset.pickle'):
    """
    :param filename: pickle name
    :return: dictionary or list
    """
    with open(filename, 'rb') as handle:
        ids = pickle.load(handle)
    return ids


def save_pickle(dict, filename='split_dataset.pickle'):
    """
    :param dict: dictionary to be saved
    :param filename: pickle name
    :return: None
    """
    with open(filename, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def normalize_min_max(nparray):
    """
    :param nparray: input img (feature)
    :return: normalized nparray
    """
    nmin = np.amin(nparray)
    nmax = np.amax(nparray)
    norm_array = (nparray - nmin) / (nmax - nmin)
    return norm_array


def combine_total_avg(output, side_len, margin):
    """
    combine all things together and average overlapping areas of prediction
    curxinfo = [[curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]...]
    : param output: list of all coordinates and voxels of sub-volumes
    : param side_len: shape of the target volume
    : param margin: stride length of sliding window
    return: output_org, combined volume, original size
    return: curorigin, origin of CT
    return: curspacing, spacing of CT
    """
    curtemp = output[0]
    curshape = curtemp[3]
    curorigin = curtemp[4]
    curspacing = curtemp[5]
    #########################################################################
    nz, nh, nw = curtemp[2][0], curtemp[2][1], curtemp[2][2]
    [z, h, w] = curshape
    if type(margin) is not list:
        margin = [margin, margin, margin]

    splits = {}
    for i in range(len(output)):
        curinfo = output[i]
        curxdata = curinfo[0]
        cursplitID = int(curinfo[1])
        if not (cursplitID in splits.keys()):
            splits[cursplitID] = curxdata
        else:
            continue  # only choose one splits

    output = np.zeros((z, h, w), np.float32)

    count_matrix = np.zeros((z, h, w), np.float32)

    idx = 0
    for iz in range(nz + 1):
        for ih in range(nh + 1):
            for iw in range(nw + 1):
                sz = iz * side_len[0]
                ez = iz * side_len[0] + margin[0]
                sh = ih * side_len[1]
                eh = ih * side_len[1] + margin[1]
                sw = iw * side_len[2]
                ew = iw * side_len[2] + margin[2]
                if ez > z:
                    sz = z - margin[0]
                    ez = z
                if eh > h:
                    sh = h - margin[1]
                    eh = h
                if ew > w:
                    sw = w - margin[2]
                    ew = w
                split = splits[idx]
                ##assert (split.shape[0] == margin[0])
                ##assert (split.shape[1] == margin[1])
                ##assert (split.shape[2] == margin[2])
                # [margin[0]:margin[0] + side_len[0], margin[1]:margin[1] + \
                # side_len[1], margin[2]:margin[2] + side_len[2]]
                output[sz:ez, sh:eh, sw:ew] += split
                count_matrix[sz:ez, sh:eh, sw:ew] += 1
                idx += 1

    output = output / count_matrix
    output_org = output
    # output_org = output[:zorg, :horg, :worg]
    ##min_value = np.amin(output_org.flatten())
    ##max_value = np.amax(output_org.flatten())
    ##assert (min_value >= 0 and max_value <= 1)
    return output_org, curorigin, curspacing


def combine_total(output, side_len, margin):
    """
    combine all things together without average overlapping areas of prediction
    curxinfo = [[curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]...]
    : param output: list of all coordinates and voxels of sub-volumes
    : param side_len: shape of the target volume
    : param margin: stride length of sliding window
    return: output_org, combined volume, original size
    return: curorigin, origin of CT
    return: curspacing, spacing of CT
    """
    curtemp = output[0]
    curshape = curtemp[3]
    curorigin = curtemp[4]
    curspacing = curtemp[5]

    nz, nh, nw = curtemp[2][0], curtemp[2][1], curtemp[2][2]
    [z, h, w] = curshape
    #### output should be sorted
    if type(margin) is not list:
        margin = [margin, margin, margin]
    splits = {}
    for i in range(len(output)):
        curinfo = output[i]
        curxdata = curinfo[0]
        cursplitID = int(curinfo[1])
        splits[cursplitID] = curxdata

    output = -1000000 * np.ones((z, h, w), np.float32)

    idx = 0
    for iz in range(nz + 1):
        for ih in range(nh + 1):
            for iw in range(nw + 1):
                sz = iz * side_len[0]
                ez = iz * side_len[0] + margin[0]
                sh = ih * side_len[1]
                eh = ih * side_len[1] + margin[1]
                sw = iw * side_len[2]
                ew = iw * side_len[2] + margin[2]
                if ez > z:
                    sz = z - margin[0]
                    ez = z
                if eh > h:
                    sh = h - margin[1]
                    eh = h
                if ew > w:
                    sw = w - margin[2]
                    ew = w
                split = splits[idx]
                ##assert (split.shape[0] == margin[0])
                ##assert (split.shape[1] == margin[1])
                ##assert (split.shape[2] == margin[2])
                output[sz:ez, sh:eh, sw:ew] = split
                idx += 1
    output_org = output
    # output_org = output[:z, :h, :w]
    ##min_value = np.amin(output_org.flatten())
    ##assert (min_value >= -1000000)
    return output_org, curorigin, curspacing


def save_itk(image, origin, spacing, filename):
    """
    :param image: images to be saved
    :param origin: CT origin
    :param spacing: CT spacing
    :param filename: save name
    :return: None
    """
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


def load_itk_image(filename):
    """
    :param filename: CT name to be loaded
    :return: CT image, CT origin, CT spacing
    """
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def lumTrans(img):
    """
    :param img: CT image
    :return: Hounsfield Unit window clipped and normalized
    """
    lungwin = np.array([-1000., 400.])
    # the upper bound 400 is already ok
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


class Logger(object):
    """
    Logger from screen to txt file
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def dice_coef_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: dice coefficient
    """

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def ppv_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: positive predictive value
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_pred_f) + smooth)


def sensitivity_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: sensitivity
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + smooth)


def acc_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: accuracy
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    y_pred_f = y_pred_f > 0.5
    intersection = np.sum(y_true_f == y_pred_f)
    return (intersection) / (len(y_true_f) + smooth)


def debug_dataloader(train_loader, testFolder):
    """
    :param train_loader: training data for debug
    :param testFolder: save directory
    :return: None
    """
    for i, (x, y, coord, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(train_loader):
        xdata = x.numpy()
        ydata = y.numpy()
        origindata = org.numpy()
        spacingdata = spac.numpy()
        NameID = NameID[0]
        SplitID = SplitID[0]
        nzhw = nzhw.numpy()
        ShapeOrg = ShapeOrg.numpy()
        for j in range(xdata.shape[0]):
            if i < 2 and j < 5:
                curcube = xdata[j, 0] * 255
                cury = ydata[j, 0]
                curorigin = origindata[j].tolist()
                curspacing = spacingdata[j].tolist()
                curnameID = NameID[j]
                print('cursplit id ', SplitID[j])
                curpath = os.path.join(testFolder, 'test-%s-%d-%d-cube.nii.gz' % (curnameID, i, j))
                curypath = os.path.join(testFolder, 'test-%s-%d-%d-cubey.nii.gz' % (curnameID, i, j))
                curwpath = os.path.join(testFolder, 'test-%s-%d-%d-cubeweight.nii.gz' % (curnameID, i, j))
                save_itk(curcube.astype(dtype='uint8'), curorigin, curspacing, curpath)
                save_itk(cury.astype(dtype='uint8'), curorigin, curspacing, curypath)
    return

def make_dir(path):
    if not os.path.exists(path):
        print('make dir:', path)
        os.mkdir(path)


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




def crop(src, tar):
    z = src.shape[2] - tar.shape[2]
    y = src.shape[3] - tar.shape[3]
    x = src.shape[4] - tar.shape[4]
    src = src[:,:,int(z//2):int(src.shape[2]-z//2),
          int(y//2):int(src.shape[3]-y//2),
          int(x//2):int(src.shape[4]-x//2)]
    return src