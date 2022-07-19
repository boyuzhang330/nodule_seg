#!/usr/bin/env python  
# encoding: utf-8  
import argparse
import numpy as np

config = {'pad_value': 0,
          'augtype': {'flip': True, 'swap': False, 'smooth': False, 'jitter': False, 'split_jitter': True},
          'startepoch': 0,
          'lr_stage': np.array([10, 20, 30, 1000000]),
          # 'lr': np.array([3e-3, 3e-4, 3e-5, 3e-6]),
          'lr': np.array([0.01, 0.001, 0.0001, 0.00001]),
          # 'dataset_path': '/home/zhangboyu/dataset/Artery/crop_data',
          'dataset_path': r'E:\work\dataSet\nodule_data\crop_data',
          # 'dataset_split': '/home/zhangboyu/nodule_seg/other/split_artery.pickle',
          'dataset_split': r'D:\work\project\nodule_seg\other/split_artery.pickle',
          'suoxiao': 0,
          'artery': 1,
          'Vnet':0,
          'Vnet_3':0,
          'boundary_aware':1.
          }

parser = argparse.ArgumentParser(description='PyTorch Airway Segmentation')
parser.add_argument('--model', '-m', metavar='MODEL', default='baseline', help='model')
parser.add_argument('--out-channels', '-c', default=2, type=int, metavar='N', help='number of output channels')
# parser.add_argument('--save-dir', default='/home/zhangboyu/nodule_seg/working/save_artery_OHEM', type=str, metavar='SAVE', help='directory to save checkpoint (default: none)')
parser.add_argument('--save-dir', default='D:\work\project\working\save_test', type=str, metavar='SAVE',help='directory to save checkpoint (default: none)')
# parser.add_argument('--resume', default='D:\work\project\working\save_artery_BA/046.ckpt', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--debugval', default=0, type=int, metavar='Validation', help='debug mode for validation')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 16)')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=None, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--save-freq', default='1', type=int, metavar='S', help='save frequency')
parser.add_argument('--val-freq', default='1', type=int, metavar='S', help='validation frequency')
parser.add_argument('--test-freq', default='0', type=int, metavar='S', help='testing frequency')
parser.add_argument('--resumepart', default=0, type=int, metavar='PARTRESUME', help='Resume params. part')
parser.add_argument('--test', default=0, type=int, metavar='TEST', help='1 do test evaluation, 0 not')
parser.add_argument('--debug', default=0, type=int, metavar='TEST', help='debug mode')
parser.add_argument('--randsel', default=0, type=int, metavar='RandomSelect',
                    help='randomly select samples for training')
parser.add_argument('--featsave', default=0, type=int, metavar='FeatSave', help='save SAD features')
parser.add_argument('--sgd', default=0, type=int, metavar='SGDopti', help='use sgd')
parser.add_argument('--debugdataloader', default=0, type=int, metavar='DataDebug', help='debug mode for dataloader')
parser.add_argument('--cubesize', default=[64, 176, 176], nargs="*", type=int, metavar='cube', help='cube size')
# parser.add_argument('--cubesize', default=[64, 64, 64], nargs="*", type=int, metavar='cube', help='cube size')
# parser.add_argument('--cubesize', default=[68, 68, 68], nargs="*", type=int, metavar='cube', help='cube size')
parser.add_argument('--cubesizev', default=None, nargs="*", type=int, metavar='cube', help='cube size')
parser.add_argument('--stridet', default=[48, 48, 48], nargs="*", type=int, metavar='stride', help='split stride train')
parser.add_argument('--stridev', default=[48, 48, 48], nargs="*", type=int, metavar='stride', help='split stride val')
parser.add_argument('--multigpu', default=True, type=bool, metavar='mgpu', help='use multiple gpus')

# 优化
parser.add_argument('--OHEM', default=False, type=bool, metavar='*', help='use OHEM 1000 patch')
parser.add_argument('--OHEM_num', default=100, type=bool, metavar='*', help='use OHEM 1000 patch')


