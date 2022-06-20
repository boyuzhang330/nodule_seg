# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import time
import numpy as np
import data
from importlib import import_module
import shutil
from trainval_classifier import train_casenet, val_casenet_per_case
from utils import Logger, save_itk, weights_init, debug_dataloader,CosineAnnealingWarmupRestarts
import sys
from torch.utils.tensorboard import SummaryWriter



sys.path.append('../')
from split_combine_mj import SplitComb
from torch.nn import DataParallel
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
import csv
from nodule_option import parser, config


def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(0)
    # torch.cuda.set_device(0)
    print('----------------------Load Model------------------------')
    model = import_module(args.model)
    net = model.get_model(config,args)
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    save_dir = os.path.join('results', save_dir)
    print(args)
    print("savedir: ", save_dir)
    print("args.lr: ", args.lr)
    args.lr_stage = config['lr_stage']
    args.lr_preset = config['lr']
    print(args.lr_stage,args.lr_preset)

    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir='runs/nodule')

    if args.resume:
        resume_part = args.resumepart
        checkpoint = torch.load(args.resume)

        if resume_part:
            """
            load part of the weight parameters
            """
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            print('part load from {} Done'.format(args.resume))
        else:
            """
            load full weight parameters
            """
            net.load_state_dict(checkpoint['state_dict'])
            # net.load_state_dict(checkpoint)
            print("full resume from {} Done".format(args.resume))
    else:
        weights_init(net, init_type='xavier')  # weight initialization

    if args.epochs is None:
        end_epoch = args.lr_stage[-1]
    else:
        end_epoch = args.epochs

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log.txt')

    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))

        pyfiles = [f for f in os.listdir('./models') if f.endswith('.py')]
        if not os.path.exists(os.path.join(save_dir, "models")):
            os.makedirs(os.path.join(save_dir, "models"))
        for f in pyfiles:
            shutil.copy(os.path.join("./models", f), os.path.join(save_dir, "models", f))

    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_counts = torch.cuda.device_count()
        print('Using {} {} GPUs, Total Memory: {}Gb'.format(gpu_counts, gpu_properties.name,
                                                            gpu_counts * gpu_properties.total_memory / 1024.0 ** 3))
        net = net.cuda()
        cudnn.benchmark = True
        if args.multigpu:
            net = DataParallel(net)

    if args.cubesizev is not None:
        marginv = args.cubesizev
    else:
        marginv = args.cubesize
    print('validation stride ', args.stridev)

    # if not args.sgd:
    #     optimizer = optim.Adam(net.parameters(), lr=1e-3)  # args.lr
    # else:
    #     optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    if args.test:
        print('---------------------testing---------------------')
        # type='train'
        type='val'
        split_comber = SplitComb(args.stridev, marginv)
        dataset_test = data.NoduleDataSlow(
            config,
            phase=type,
            split_comber=split_comber,
            debug=args.debug,
            random_select=False)
        test_loader = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)
        epoch = 111
        print('start testing')

        testdata,testrecall = val_casenet_per_case(epoch, net, test_loader, args, save_dir)

        return




    print('---------------------------------Load Dataset--------------------------------')
    margin = args.cubesize
    print('patch size ', margin)
    print('train stride ', args.stridet)
    split_comber = SplitComb(args.stridet, margin)

    dataset_train = data.NoduleDataSlow(
        config,
        phase='train',
        split_comber=split_comber,
        debug=args.debug,
        random_select=args.randsel)



    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    print('--------------------------------------')
    split_comber = SplitComb(args.stridev, marginv)

    # load validation dataset
    dataset_val = data.NoduleDataSlow(
        config,
        phase='val',
        split_comber=split_comber,
        debug=args.debug,
        random_select=False)
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    print('--------------------------------------')

    # load testing dataset
    dataset_test = data.NoduleData(
    	config,
    	phase='test',
    	split_comber=split_comber,
    	debug=args.debug,
    	random_select=False)
    test_loader = DataLoader(
    	dataset_test,
    	batch_size=args.batch_size,
    	shuffle=False,
    	num_workers=args.workers,
    	pin_memory=True)


    ##############################
    # start training
    ##############################

    total_epoch = []
    train_loss = []
    val_loss = []
    test_loss = []

    train_acc = []
    val_acc = []
    test_acc = []

    train_sensi = []
    val_sensi = []
    test_sensi = []

    dice_train = []
    dice_val = []
    dice_test = []

    ppv_train = []
    ppv_val = []
    ppv_test = []

    logdirpath = os.path.join(save_dir, 'log')
    if not os.path.exists(logdirpath):
        os.mkdir(logdirpath)

    v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = 0, 0, 0, 0, 0
    te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = 0, 0, 0, 0, 0


    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=20,
                                              cycle_mult=1.0,
                                              max_lr=0.01,
                                              min_lr=0.0001,
                                              warmup_steps=5,
                                              gamma=0.5)

    # 将模型写入tensorboard
    # init_img = torch.zeros((1,1,64,64,64),device='cuda')
    # tb_writer.add_graph(net.module,init_img)

    for epoch in range(start_epoch, end_epoch + 1):
        print("Training epoch: {}".format(epoch))
        mean_loss, mean_dice, mean_acc, mean_sensiti, mean_ppv= train_casenet(epoch, net, train_loader, optimizer, args,save_dir,scheduler)


        train_loss.append(mean_loss)
        dice_train.append(mean_dice)


        # Save the current model
        # if args.multigpu:
        #     state_dict = net.module.state_dict()
        # else:
        #     state_dict = net.state_dict()
        # for key in state_dict.keys():
        #     state_dict[key] = state_dict[key].cpu()
        # torch.save({
        #     'state_dict': state_dict,
        #     'args': args},
        #     os.path.join(save_dir, 'latest.ckpt'))

        # Save the model frequently
        if epoch % args.save_freq == 0:
            if args.multigpu:
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
        # 如果需要validation可以取消注释，但validation需要消耗很大的内存，建议单独起实验自行开启
        if (epoch % args.val_freq == 0) or (epoch == start_epoch):
            print(save_dir)
            print("Validation epoch: {}".format(epoch))
            val_dice,val_recall = val_casenet_per_case(epoch, net, val_loader, args, save_dir)
            print("Validation epoch: {} dice:{}".format(epoch,val_dice))

        # 训练结果写入tensorboard


        tb_writer.add_scalar('train_loss',mean_loss,epoch)
        tb_writer.add_scalar('train_dice',mean_dice,epoch)
        tb_writer.add_scalar('train_acc',mean_acc,epoch)
        tb_writer.add_scalar('train_sensiti',mean_sensiti,epoch)
        tb_writer.add_scalar('train_ppv',mean_ppv,epoch)
        tb_writer.add_scalar('test_dice',val_dice,epoch)
        tb_writer.add_scalar('test_recall',val_recall,epoch)
        tb_writer.add_scalar('learning_rate',optimizer.state_dict()['param_groups'][0]['lr'],epoch)



        val_loss.append(v_loss)
        val_acc.append(mean_acc2)
        val_sensi.append(mean_sensiti2)
        dice_val.append(mean_dice2)
        ppv_val.append(mean_ppv2)

        test_loss.append(te_loss)
        test_acc.append(mean_acc3)
        test_sensi.append(mean_sensiti3)
        dice_test.append(mean_dice3)
        ppv_test.append(mean_ppv3)

        total_epoch.append(epoch)

        totalinfo = np.array([total_epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc,
                              train_sensi, val_sensi, test_sensi, dice_train, dice_val, dice_test,
                              ppv_train, ppv_val, ppv_test])
        np.save(os.path.join(logdirpath, 'log.npy'), totalinfo)

    logName = os.path.join(logdirpath, 'log.csv')
    with open(logName, 'a') as csvout:
        writer = csv.writer(csvout)
        row = ['train epoch', 'train loss', 'val loss', 'test loss', 'train acc', 'val acc', 'test acc',
               'train sensi', 'val sensi', 'test sensi', 'dice train', 'dice val', 'dice test',
               'ppv train', 'ppv val', 'ppv test']
        writer.writerow(row)

        for i in range(len(total_epoch)):
            row = [total_epoch[i], train_loss[i], val_loss[i], test_loss[i],
                   train_acc[i], val_acc[i], test_acc[i],
                   train_sensi[i], val_sensi[i], test_sensi[i],
                   dice_train[i], dice_val[i], dice_test[i],
                   ppv_train[i], ppv_val[i], ppv_test[i]]
            writer.writerow(row)
        csvout.close()

    print("Done")
    return


if __name__ == '__main__':
    main()
