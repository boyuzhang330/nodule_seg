import numpy as np
import pickle
import os
import time
import torch
import shutil
import torch.nn.functional as F
from collections import OrderedDict
from data import OHEM_dataset
from split_combine_mj import SplitComb
from utils import save_itk, load_itk_image, dice_coef_np, ppv_np, \
    sensitivity_np, acc_np, combine_total_avg, combine_total, normalize_min_max,CosineAnnealingWarmupRestarts,crop
from loss import dice_loss, binary_cross_entropy, focal_loss, sad_loss,general_union_loss,weighted_softmax_cross_entropy_with_logits_ignore_labels
from torch.cuda import empty_cache
import csv
from scipy.ndimage.interpolation import zoom
import warnings
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

th_bin = 0.5


def function(date):
    # print(date[-1])
    return date[-1]

def get_lr(epoch, args):
    """
    :param epoch: current epoch number
    :param args: global arguments args
    :return: learning rate of the next epoch
    """
    if args.lr is None:
        assert epoch <= args.lr_stage[-1]
        lrstage = np.sum(epoch > args.lr_stage)
        lr = args.lr_preset[lrstage]
    else:
        lr = args.lr
    return lr


def train_casenet(epoch, model, data_loader, optimizer, args, save_dir,scheduler,conf):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: training data
    :param optimizer: training optimizer
    :param args: global arguments args
    :param save_dir: save directory
    :return: performance evaluation of the current epoch
    """
    model.train()
    starttime = time.time()
    sidelen = args.stridet
    margin = args.cubesize

    # lr = get_lr(epoch, args)

    # print(scheduler)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = scheduler
    # assert (lr is not None)
    optimizer.zero_grad()

    lossHist = []
    dice_total = {}
    ppv_total = {}
    acc_total = {}
    dice_hard_total = {}
    sensitivity_total = {}

    traindir = os.path.join(save_dir, 'train')
    if not os.path.exists(traindir):
        os.mkdir(traindir)
    training_log = os.path.join(traindir, 'train_log.txt')
    f = open(training_log, 'w')
    loss = 0.0
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    for i, (x, y, coord, org, spac, NameID, SplitID, nzhw, ShapeOrg,boundary) in enumerate(data_loader):
        torch.cuda.empty_cache()
        if i % int(len(data_loader) / 10 + 1) == 0:
            print("Epoch: {}, Iteration: {}/{}, loss: {}".format(epoch, i, len(data_loader), loss))
        batchlen = x.size(0)

        x = x.cuda()
        y = y.cuda()

        ###############################
        if conf['boundary_aware']==1:
            casePred,boundary_pred = model(x, coord)
        else:
            casePred = model(x,coord)

        if casePred.shape!=y.shape:
            y = crop(y,casePred)
        # loss = F.cross_entropy(casePred,torch.squeeze(y, dim=1).long())
        loss = dice_loss(casePred, y)
        loss += focal_loss(casePred, y)

        if conf['boundary_aware']==1:
            boundary = boundary.cuda()
            loss += weighted_softmax_cross_entropy_with_logits_ignore_labels(boundary_pred, boundary, 1000)

        if epoch == 1 and i == 0:
            print("Input X shape: {}, Y shape: {}, Output num: {}, Output Shape: {}".format(x.shape, y.shape,
                                                                                  len(casePred),
                                                                                                casePred.shape))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossHist.append(loss.item())


        if epoch % args.save_freq == 0:
            casePred = F.softmax(casePred, dim=1)
            outdata = casePred.cpu().data.numpy()
            segdata = y.cpu().data.numpy().sum(axis=1, keepdims=True)
            outdatabw = np.argmax(outdata, axis=1)  #
            for j in range(batchlen):
                # multi-category artery/vein classification
                for curcategory in range(1, casePred.shape[1]):
                    segpred = (outdatabw[j] == curcategory)
                    segpred = segpred.astype('float')
                    curgt = (segdata == curcategory)
                    dice = dice_coef_np(outdata[j, curcategory], curgt[j, 0])
                    ppv = ppv_np(segpred, curgt[j, 0])
                    sensiti = sensitivity_np(segpred, curgt[j, 0])
                    acc = acc_np(segpred, curgt[j, 0])
                    dicehard = dice_coef_np(segpred, curgt[j, 0])
                    ##########################################################################
                    if not (curcategory in dice_total):
                        dice_total[curcategory] = []
                    dice_total[curcategory].append(dice)
                    if not (curcategory in ppv_total):
                        ppv_total[curcategory] = []
                    ppv_total[curcategory].append(ppv)
                    if not (curcategory in sensitivity_total):
                        sensitivity_total[curcategory] = []
                    sensitivity_total[curcategory].append(sensiti)
                    if not (curcategory in acc_total):
                        acc_total[curcategory] = []
                    acc_total[curcategory].append(acc)
                    if not (curcategory in dice_hard_total):
                        dice_hard_total[curcategory] = []
                    dice_hard_total[curcategory].append(dicehard)

            if i <= 3:
                print("batch time: {}".format(time.time() - starttime))

    if args.OHEM ==True:
        print()
        OHEM_path = 'OHEM_save'
        if os.path.exists(OHEM_path):
            shutil.rmtree(OHEM_path)
        os.mkdir(OHEM_path)
        print('Start OHEM: ')
        OHEM_list=[]

        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for i, (x, y, coord, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(data_loader):
                if i <= 3:
                    print("batch time: {}".format(time.time() - starttime))

                x = x.cuda()
                y = y
                casePred = model(x, coord)

                for num,pred_one in enumerate(casePred): # [2,2,64,64,64]
                    pred_one = F.softmax(pred_one,dim=0)
                    pred_one = np.argmax(pred_one.cpu().data.numpy(),axis=0)

                    y_one = y[num].cpu().data.numpy()
                    one_loss = dice_coef_np(pred_one, y_one) # 通过dice loss找到难样本


                    # print(x[num].shape)
                    # print(y[num].shape)
                    # print(coord[num].shape)
                    # print(one_loss)
                    # print(NameID[0][num])
                    # print(SplitID[0][num])
                    origin = org[num]
                    spacing = spac[num].data.numpy()
                    # print(origin)
                    # print(spacing)
                    # print(len(spacing))

                    name=NameID[0][num].split('/')[-1]+'_'+str(SplitID[0][num].cpu().data.numpy())
                    # print(name)
                    save_itk(x[num][0].cpu().data.numpy(),origin,spacing,os.path.join(OHEM_path,name+'_x.nii.gz'))
                    save_itk(y[num][0].cpu().data.numpy(),origin,spacing,os.path.join(OHEM_path,name+'_y.nii.gz'))
                    save_itk(coord[num].cpu().data.numpy(),[1,1,1,1],[1,1,1,1],os.path.join(OHEM_path,name+'_coord.nii.gz'))
                    # np.save(os.path.join(OHEM_path,name+'_coord.npy'),coord[num].cpu().data.numpy())
                    e = [name,one_loss]
                    OHEM_list.append(e)

                    # input('stop')



            OHEM_list.sort(reverse=True,key=function)
            if args.OHEM_num < len(OHEM_list):
                OHEM_list = OHEM_list[:args.OHEM_num]

        model.train()
        optimizer.zero_grad()
        print('已找到最难识别的{}个样本'.format(len(OHEM_list)))

        dataset_OHEM = OHEM_dataset(OHEM_list,OHEM_path)
        OHEM_loader = DataLoader(
            dataset_OHEM,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)

        for i, (x,y,coord) in enumerate(OHEM_loader):
            torch.cuda.empty_cache()

            x = x.cuda()
            y = y.cuda()
            casePred = model(x, coord)
            loss = dice_loss(casePred, y)
            loss += focal_loss(casePred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('OHEM over')

    scheduler.step()

    endtime = time.time()
    lossHist = np.array(lossHist)
    mean_loss = np.sum(lossHist)

    mean_dice = {}
    mean_dice_hard = {}
    mean_ppv = {}
    mean_sensiti = {}
    mean_acc = {}

    mean_dice_all = 0
    mean_dice_hard_all = 0
    mean_ppv_all = 0
    mean_sensiti_all = 0
    mean_acc_all = 0

    for curkey in dice_total.keys():
        curdice = np.mean(np.array(dice_total[curkey]))
        mean_dice[curkey] = curdice
        mean_dice_all += curdice

        curdicehard = np.mean(np.array(dice_hard_total[curkey]))
        mean_dice_hard[curkey] = curdicehard
        mean_dice_hard_all += curdicehard

        curppv = np.mean(np.array(ppv_total[curkey]))
        mean_ppv[curkey] = curppv
        mean_ppv_all += curppv

        cursensi = np.mean(np.array(sensitivity_total[curkey]))
        mean_sensiti[curkey] = cursensi
        mean_sensiti_all += cursensi

        curacc = np.mean(np.array(acc_total[curkey]))
        mean_acc[curkey] = curacc
        mean_acc_all += curacc

    mean_dice_all /= float(casePred.shape[1] - 1)
    mean_dice_hard_all /= float(casePred.shape[1] - 1)
    mean_ppv_all /= float(casePred.shape[1] - 1)
    mean_sensiti_all /= float(casePred.shape[1] - 1)
    mean_acc_all /= float(casePred.shape[1] - 1)

    print(
        'Train, epoch %d, loss %.4f, accuracy %.4f, sensitivity %.4f, dice %.4f, dice hard %.4f, ppv %.4f, time %3.2f, lr % .5f '
        % (epoch, mean_loss, mean_acc_all, mean_sensiti_all, mean_dice_all, mean_dice_hard_all, mean_ppv_all,
           endtime - starttime, lr))


    empty_cache()

    return mean_loss, mean_dice_all, mean_acc_all, mean_sensiti_all, mean_ppv_all



def val_casenet_per_case(epoch, model, data_loader, args, save_dir,conf):
    """
    :param epoch: current epoch number
    :param model: CNN model
    :param data_loader: evaluation and testing data
    :param args: global arguments args
    :param save_dir: save directory
    :param test_flag: current mode of validation or testing
    :return: performance evaluation of the current epoch
    """
    model.eval()
    starttime = time.time()

    sidelen = args.stridev
    if args.cubesizev is not None:
        margin = args.cubesizev
    else:
        margin = args.cubesize

    name_total = []
    lossHist = []

    # dice_total = []
    # ppv_total = []
    # sensitivity_total = []
    # dice_hard_total = []
    # acc_total = []


    valdir = os.path.join(save_dir,'val%03d'%(epoch))
    if not os.path.exists(valdir):
        os.mkdir(valdir)

    dice_total = {}
    recall_total = {}
    all_dice = 0
    all_recall = 0

    print(len(data_loader))
    with torch.no_grad():
        for data_name, data in data_loader.dataset.allimgdata_memory.items():

            case_cube = []
            batch = []
            # 102171_T2 D:\work\dataSet\nodule_data\crop_data\102171_T2.nii
            data_arr,origin,spacing=load_itk_image(data)
            dcm_img = data_arr
            orgshape = dcm_img.shape

            curorigin = origin
            curspacing = spacing
            labels = data_loader.dataset.alllabeldata_memory[data_name]

            labels,_,_=load_itk_image(labels)
            split_coord = []
            [z, h, w] = orgshape
            output = np.zeros((z, h, w), np.float32)

            for cube in data_loader.dataset.cubelist:
                if cube[0] != data_name:
                    continue
                case_cube.append(cube)

            for i in range(len(case_cube)):

                cursplit = case_cube[i][1]
                curcube = dcm_img[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                          cursplit[2][0]:cursplit[2][1]]
                curcube = (curcube.astype(np.float32)) / 255.0

                # calculate the coordinate for coordinate-aware convolution
                start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
                normstart = ((np.array(start).astype('float') / np.array(orgshape).astype('float')) - 0.5) * 2.0
                crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
                stride = 1.0
                normsize = (np.array(crop_size).astype('float') / np.array(orgshape).astype('float')) * 2.0
                xx, yy, zz = np.meshgrid(
                    np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
                    np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
                    np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
                    indexing='ij')
                coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]],
                                       0).astype(
                    'float')
                curcube = curcube[np.newaxis, ...]
                batch.append([curcube, coord])
                split_coord.append(cursplit)
                if len(batch) < 5 and i != len(case_cube) - 1:
                    continue

                input_cube = torch.from_numpy(np.array([batch[i][0] for i in range(len(batch))])).float()
                input_coord = torch.from_numpy(np.array([batch[i][1] for i in range(len(batch))])).float()

                 # if torch.cuda.is_available():
                #     input_cube = input_cube.cuda(config.DATA.GPU_ID)
                #     input_coord = input_coord.cuda(config.DATA.GPU_ID)

                casePreds = model(input_cube,input_coord)
                # casePreds = model(input_cube)
                if len(casePreds)==2:
                    casePred=casePreds[0]
                else:

                    casePred = casePreds

                preds = F.softmax(casePred, dim=1)
                # 第一种方法
                outdatabw = np.argmax(preds.cpu().data.numpy(), axis=1)  # (B, D, H, W)
                # 第二种方法 置信度
                # preds=preds[:,1,:,:,:]
                # preds[preds>zhixin]=1
                # outdatabw=preds.cpu().data.numpy()
                for k in range(len(batch)):
                    cursplit = split_coord[k]
                    try:
                        output[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                        cursplit[2][0]:cursplit[2][1]] = outdatabw[k]
                    except:
                        output[cursplit[0][0]+17:cursplit[0][1]-17, cursplit[1][0]+17:cursplit[1][1]-17,
                        cursplit[2][0]+17:cursplit[2][1]-17] = outdatabw[k]

                batch = []
                split_coord = []

            torch.cuda.empty_cache()

            p_combine_bw = output
            y_combine = labels

            curypath = os.path.join(valdir, '%s-gt.nii.gz' % (data_name.split('/')[-1]))
            curpredpath = os.path.join(valdir, '%s-pred.nii.gz' % (data_name.split('/')[-1]))
            save_itk(y_combine.astype(dtype='uint8'), curorigin, curspacing, curypath)
            save_itk(p_combine_bw.astype(dtype='uint8'), curorigin, curspacing, curpredpath)

            dice_total[data_name] = []
            for i in range(1, casePred.shape[1]):
                pbw = p_combine_bw == i
                gbw = y_combine == i
                dice_total[data_name].append(dice_coef_np(pbw, gbw))
            recall_total[data_name]=[]
            recall_total[data_name].append(sensitivity_np(p_combine_bw,y_combine))
            ########################################################################
            mean_dice = np.mean(dice_total[data_name])
            mean_recall = np.mean(recall_total[data_name])
            dice_total[data_name].append(mean_dice)
            # print("{}, Dice: {}".format(data_name, mean_dice))
            all_dice+=mean_dice
            all_recall+=mean_recall
            with open(os.path.join(valdir, 'val_results.csv'), 'a') as csvout:
                writer = csv.writer(csvout)
                row = ['name:',data_name,'dice:',mean_dice]
                writer.writerow(row)
            del y_combine, p_combine_bw
            # except:
            #     print(data_name,'error')

    return all_dice/len(dice_total),all_recall/len(dice_total)