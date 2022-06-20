import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
import SimpleITK as sitk
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from utils import save_itk, load_itk_image, lumTrans, load_pickle


class NoduleData(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, config, phase='train', split_comber=None,
                 debug=False, random_select=False):
        """
        :param config: configuration from model
        :param phase: training or validation or testing
        :param split_comber: split-combination-er
        :param debug: debug mode to check few data
        :param random_select: use partly, randomly chosen data for training
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.augtype = config['augtype']
        self.split_comber = split_comber
        self.rand_sel = random_select
        self.patch_per_case = 5  # patches used per case if random training
        self.debug_flag = debug
        """
        specify the path and data split
        """
        self.datapath = config['dataset_path']
        self.dataset = load_pickle(config['dataset_split'])
        print("Load data record from {}".format(config["dataset_split"]))

        print("-------------------------Load all data into memory---------------------------")
        """
        count the number of cases
        """
        labellist = []
        cubelist = []
        self.caseNumber = 0
        allimgdata_memory = {}
        alllabeldata_memory = {}

        if self.phase == 'train':
            file_names = self.dataset['train']
            data_file_names = file_names['lidc']
            file_num = len(data_file_names)
            if self.debug_flag:
                data_file_names = file_names['lidc'][:3]
                file_num = len(data_file_names)
            self.caseNumber += file_num

            print("total %s case number: %d" % (self.phase, self.caseNumber))
            for raw_path in data_file_names:
                raw_path = os.path.join(self.datapath, raw_path.split('/')[-1])
                assert (os.path.exists(raw_path) is True)
                label_path = raw_path.replace('.nii.gz', '_label.nii.gz')
                assert (os.path.exists(label_path) is True)

                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)

                data_name = raw_path.split('\\')[-1].split('.')[0]

                print("Name: %s, # of splits: %d" % (data_name, len(splits)))
                labels, _, _ = load_itk_image(label_path)

                allimgdata_memory[data_name] = [imgs, origin, spacing]
                alllabeldata_memory[data_name] = labels
                cube_train = []
                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]
                    labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                                cursplit[2][0]:cursplit[2][1]]
                    if config['suoxiao']==1:
                        curnumlabel = np.sum(labelcube[17:-17,17:-17,17:-17])
                        print(curnumlabel)
                    else:
                        curnumlabel = np.sum(labelcube)


                    labellist.append(curnumlabel)
                    # filter out those zero-0 labels
                    if curnumlabel > 0 and np.sum(np.array(labelcube.shape) >= np.array(self.split_comber.side_len)) == len(labelcube.shape):
                        curlist = [data_name, cursplit, j, nzhw, orgshape, 'Y']
                        cube_train.append(curlist)

                random.shuffle(cube_train)

                if self.rand_sel:
                    """
                    only chooses random number of patches for training
                    """
                    cubelist.append(cube_train)
                else:
                    cubelist += cube_train
        elif self.phase == 'val':
            file_names = self.dataset['val']
            data_file_names = file_names['lidc']
            file_num = len(data_file_names)
            if self.debug_flag:
                data_file_names = file_names['lidc'][:1]
                file_num = len(data_file_names)

            self.caseNumber += file_num
            print("total %s case number: %d" % (self.phase, self.caseNumber))

            for raw_path in data_file_names:
                raw_path = os.path.join(self.datapath, raw_path.split('/')[-1])
                assert (os.path.exists(raw_path) is True)

                label_path = raw_path.replace('.nii', '_label.nii')
                assert (os.path.exists(label_path) is True)

                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)
                data_name = raw_path.split('\\')[-1].split('.')[0]
                print("Name: %s, # of splits: %d" % (data_name, len(splits)))
                labels, _, _ = load_itk_image(label_path)

                allimgdata_memory[data_name] = [imgs, origin, spacing]
                alllabeldata_memory[data_name] = labels

                for j in range(len(splits)):
                    cursplit = splits[j]
                    curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
                    cubelist.append(curlist)

        else:
            file_names = self.dataset['test']
            data_file_names = file_names['lidc']
            file_num = len(data_file_names)
            if self.debug_flag:
                data_file_names = file_names['lidc'][:1]
                file_num = len(data_file_names)

            self.caseNumber += file_num
            print("total %s case number: %d" % (self.phase, self.caseNumber))

            for raw_path in data_file_names:
                # raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
                raw_path = os.path.join(self.datapath, raw_path.split('/')[-1])
                print(raw_path)
                assert (os.path.exists(raw_path) is True)

                label_path = raw_path.replace('clean_hu', 'label')
                assert (os.path.exists(label_path) is True)

                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)
                data_name = raw_path.split('/')[-1].split('_clean_hu')[0]
                print("Name: %s, # of splits: %d" % (data_name, len(splits)))
                labels, _, _ = load_itk_image(label_path)

                allimgdata_memory[data_name] = [imgs, origin, spacing]
                alllabeldata_memory[data_name] = labels

                for j in range(len(splits)):
                    """
                    check if this cube is suitable
                    """
                    cursplit = splits[j]
                    curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
                    cubelist.append(curlist)

        self.allimgdata_memory = allimgdata_memory
        self.alllabeldata_memory = alllabeldata_memory

        if self.rand_sel and self.phase == 'train':
            assert (len(cubelist) == self.caseNumber)
            mean_labelnum = np.mean(np.array(labellist))
            print('mean label number: %d' % (mean_labelnum))
            print('total patches: ', self.patch_per_case * self.caseNumber)

        self.cubelist = cubelist

        print('---------------------Initialization Done---------------------')
        print('Phase: %s total cubelist number: %d' % (self.phase, len(self.cubelist)))
        print()

    def __len__(self):
        """
        :return: length of the dataset
        """
        if self.phase == 'train' and self.rand_sel:
            return self.patch_per_case * self.caseNumber
        else:
            return len(self.cubelist)

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: wrapped data tensor and name, shape, origin, etc.
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        if self.phase == 'train' and self.rand_sel:
            print('rand_seg')
            caseID = idx // self.patch_per_case
            caseSplit = self.cubelist[caseID]
            np.random.shuffle(caseSplit)
            curlist = caseSplit[0]
        else:
            curlist = self.cubelist[idx]

        # train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
        # val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

        curNameID = curlist[0]
        cursplit = curlist[1]
        curSplitID = curlist[2]
        curnzhw = curlist[3]
        curShapeOrg = curlist[4]
        curtransFlag = curlist[5]
        if self.phase == 'train' and curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
            # random jittering during the training
            cursplit = augment_split_jittering(cursplit, curShapeOrg)
        ####################################################################
        imginfo = self.allimgdata_memory[curNameID]
        imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
        curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
        curcube = (curcube.astype(np.float32)) / 255.0
        ####################################################################

        # calculate the coordinate for coordinate-aware convolution
        start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
        normstart = ((np.array(start).astype('float') / np.array(curShapeOrg).astype('float')) - 0.5) * 2.0
        crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
        stride = 1.0
        normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) * 2.0
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
                                 indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
        assert (coord.shape[0] == 3)

        label = self.alllabeldata_memory[curNameID]
        # label = (label > 0)
        label = label.astype('float')
        label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]

        ####################################################################
        curNameID = [curNameID]
        curSplitID = [curSplitID]
        curnzhw = np.array(curnzhw)
        curShapeOrg = np.array(curShapeOrg)
        #######################################################################
        ########################Data augmentation##############################
        if self.phase == 'train' and curtransFlag == 'Y':
            curcube, label, coord = augment(curcube, label, coord,
                                            ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
                                            ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])

        curcube = curcube[np.newaxis, ...]
        label = label[np.newaxis, ...]
        # print(curNameID, curSplitID, curcube.shape, label.shape)

        return torch.from_numpy(curcube).float(), torch.from_numpy(label).float(), \
               torch.from_numpy(coord).float(), torch.from_numpy(origin), \
               torch.from_numpy(spacing), curNameID, curSplitID, \
               torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)





class NoduleDataSlow(Dataset):
    """
    Generate dataloader
    """

    def __init__(self, config, phase='train', split_comber=None,
                 debug=False, random_select=False):
        """
        :param config: configuration from model
        :param phase: training or validation or testing
        :param split_comber: split-combination-er
        :param debug: debug mode to check few data
        :param random_select: use partly, randomly chosen data for training
        """
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.augtype = config['augtype']
        self.split_comber = split_comber
        self.rand_sel = random_select
        self.patch_per_case = 5  # patches used per case if random training
        self.debug_flag = debug
        """
        specify the path and data split
        """
        self.datapath = config['dataset_path']
        self.dataset = load_pickle(config['dataset_split'])
        print("Load data record from {}".format(config["dataset_split"]))

        print("-------------------------Load all data into memory---------------------------")
        """
        count the number of cases
        """
        labellist = []
        cubelist = []
        self.caseNumber = 0
        allimgdata_memory = {}
        alllabeldata_memory = {}

        if self.phase == 'train':
            file_names = self.dataset['train']
            data_file_names = file_names['lidc']
            file_num = len(data_file_names)
            if self.debug_flag:
                data_file_names = file_names['lidc'][:3]
                file_num = len(data_file_names)
            self.caseNumber += file_num

            print("total %s case number: %d" % (self.phase, self.caseNumber))

            for raw_path in data_file_names:

                raw_path = os.path.join(self.datapath, raw_path.split('/')[-1])
                assert (os.path.exists(raw_path) is True)
                label_path = raw_path.replace('.nii', '_label.nii')
                assert (os.path.exists(label_path) is True)

                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)

                data_name = raw_path.split('\\')[-1].split('.')[0]

                print("Name: %s, # of splits: %d" % (data_name, len(splits)))
                labels, _, _ = load_itk_image(label_path)


                # 内存不足，存数组转为存地址。仅训练时使用。
                # allimgdata_memory[data_name] = [imgs, origin, spacing]
                # alllabeldata_memory[data_name] = labels
                allimgdata_memory[data_name] = raw_path
                alllabeldata_memory[data_name] = label_path
                cube_train = []

                for j in range(len(splits)):
                    """
                    check if this sub-volume cube is suitable
                    """
                    cursplit = splits[j]
                    labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
                                cursplit[2][0]:cursplit[2][1]]
                    if config['suoxiao']==1:
                        curnumlabel = np.sum(labelcube[17:-17,17:-17,17:-17])
                    else:
                        curnumlabel = np.sum(labelcube)
                    labellist.append(curnumlabel)
                    # filter out those zero-0 labels
                    if curnumlabel > 0 and np.sum(np.array(labelcube.shape) >= np.array(self.split_comber.side_len)) == len(labelcube.shape):
                        curlist = [data_name, cursplit, j, nzhw, orgshape, 'Y']
                        cube_train.append(curlist)

                random.shuffle(cube_train)

                if self.rand_sel:
                    """
                    only chooses random number of patches for training
                    """
                    cubelist.append(cube_train)
                else:
                    cubelist += cube_train
        elif self.phase == 'val':
            file_names = self.dataset['val']
            data_file_names = file_names['lidc']
            file_num = len(data_file_names)
            if self.debug_flag:
                data_file_names = file_names['lidc'][:1]
                file_num = len(data_file_names)

            self.caseNumber += file_num
            print("total %s case number: %d" % (self.phase, self.caseNumber))

            for raw_path in data_file_names:
                raw_path = os.path.join(self.datapath, raw_path.split('/')[-1])
                print(raw_path)
                assert (os.path.exists(raw_path) is True)
                label_path = raw_path.replace('.nii', '_label.nii')
                assert (os.path.exists(label_path) is True)

                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)
                data_name = raw_path.split('\\')[-1].split('.')[0]
                print("Name: %s, # of splits: %d" % (data_name, len(splits)))
                labels, _, _ = load_itk_image(label_path)

                # allimgdata_memory[data_name] = [imgs, origin, spacing]
                # alllabeldata_memory[data_name] = labels
                allimgdata_memory[data_name] = raw_path
                alllabeldata_memory[data_name] = label_path
                for j in range(len(splits)):
                    cursplit = splits[j]
                    curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
                    cubelist.append(curlist)

        else:
            file_names = self.dataset['test']
            data_file_names = file_names['lidc']
            file_num = len(data_file_names)
            if self.debug_flag:
                data_file_names = file_names['lidc'][:1]
                file_num = len(data_file_names)

            self.caseNumber += file_num
            print("total %s case number: %d" % (self.phase, self.caseNumber))

            for raw_path in data_file_names:
                raw_path = os.path.join(self.datapath, raw_path.split('/')[-1])
                assert (os.path.exists(raw_path) is True)

                label_path = raw_path.replace('clean_hu', 'label')
                assert (os.path.exists(label_path) is True)

                imgs, origin, spacing = load_itk_image(raw_path)
                splits, nzhw, orgshape = self.split_comber.split_id(imgs)
                data_name = raw_path.split('/')[-1].split('_clean_hu')[0]
                print("Name: %s, # of splits: %d" % (data_name, len(splits)))
                labels, _, _ = load_itk_image(label_path)

                allimgdata_memory[data_name] = [imgs, origin, spacing]
                alllabeldata_memory[data_name] = labels

                for j in range(len(splits)):
                    """
                    check if this cube is suitable
                    """
                    cursplit = splits[j]
                    curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
                    cubelist.append(curlist)

        self.allimgdata_memory = allimgdata_memory
        self.alllabeldata_memory = alllabeldata_memory

        if self.rand_sel and self.phase == 'train':
            assert (len(cubelist) == self.caseNumber)
            mean_labelnum = np.mean(np.array(labellist))
            print('mean label number: %d' % (mean_labelnum))
            print('total patches: ', self.patch_per_case * self.caseNumber)

        self.cubelist = cubelist

        print('---------------------Initialization Done---------------------')
        print('Phase: %s total cubelist number: %d' % (self.phase, len(self.cubelist)))
        print()

    def __len__(self):
        """
        :return: length of the dataset
        """
        if self.phase == 'train' and self.rand_sel:
            return self.patch_per_case * self.caseNumber
        else:
            return len(self.cubelist)

    def __getitem__(self, idx):
        """
        :param idx: index of the batch
        :return: wrapped data tensor and name, shape, origin, etc.
        """
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        if self.phase == 'train' and self.rand_sel:
            print('rand_seg')
            caseID = idx // self.patch_per_case
            caseSplit = self.cubelist[caseID]
            np.random.shuffle(caseSplit)
            curlist = caseSplit[0]
        else:
            curlist = self.cubelist[idx]

        # train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
        # val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

        curNameID = curlist[0]
        cursplit = curlist[1]
        curSplitID = curlist[2]
        curnzhw = curlist[3]
        curShapeOrg = curlist[4]
        curtransFlag = curlist[5]
        if self.phase == 'train' and curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
            # random jittering during the training
            cursplit = augment_split_jittering(cursplit, curShapeOrg)
        ####################################################################
        imginfo = self.allimgdata_memory[curNameID]
        imgs, origin, spacing = load_itk_image(imginfo)# 读地址
        # imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
        curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
        # print(curcube)
        curcube = (curcube.astype(np.float32)) / 255.0
        # print(curcube)
        ####################################################################

        # calculate the coordinate for coordinate-aware convolution
        start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
        normstart = ((np.array(start).astype('float') / np.array(curShapeOrg).astype('float')) - 0.5) * 2.0
        crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
        stride = 1.0
        normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) * 2.0
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
                                 indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
        assert (coord.shape[0] == 3)

        label = self.alllabeldata_memory[curNameID]
        label,_,_ = load_itk_image(label)# 读地址
        # label = (label > 0)
        label = label.astype('float')
        label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]

        ####################################################################
        curNameID = [curNameID]
        curSplitID = [curSplitID]
        curnzhw = np.array(curnzhw)
        curShapeOrg = np.array(curShapeOrg)
        #######################################################################
        ########################Data augmentation##############################

        if self.phase == 'train' and curtransFlag == 'Y':
            curcube, label, coord = augment(curcube, label, coord,
                                            ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
                                            ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])

        curcube = curcube[np.newaxis, ...]
        label = label[np.newaxis, ...]
        # print(curNameID, curSplitID, curcube.shape, label.shape)
        return torch.from_numpy(curcube).float(), torch.from_numpy(label).float(), \
               torch.from_numpy(coord).float(), torch.from_numpy(origin), \
               torch.from_numpy(spacing), curNameID, curSplitID, \
               torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)


def augment_split_jittering(cursplit, curShapeOrg):
    # orgshape [z, h, w]
    zstart, zend = cursplit[0][0], cursplit[0][1]
    hstart, hend = cursplit[1][0], cursplit[1][1]
    wstart, wend = cursplit[2][0], cursplit[2][1]
    curzjitter, curhjitter, curwjitter = 0, 0, 0
    if zend - zstart <= 3:
        jitter_range = (zend - zstart) * 32
    else:
        jitter_range = (zend - zstart) * 2
    # print("jittering range ", jitter_range)
    jitter_range_half = jitter_range // 2

    t = 0
    while t < 10:
        if zstart == 0:
            curzjitter = int(np.random.rand() * jitter_range)
        elif zend == curShapeOrg[0]:
            curzjitter = -int(np.random.rand() * jitter_range)
        else:
            curzjitter = int(np.random.rand() * jitter_range) - jitter_range_half
        t += 1
        if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
            break

    t = 0
    while t < 10:
        if hstart == 0:
            curhjitter = int(np.random.rand() * jitter_range)
        elif hend == curShapeOrg[1]:
            curhjitter = -int(np.random.rand() * jitter_range)
        else:
            curhjitter = int(np.random.rand() * jitter_range) - jitter_range_half
        t += 1
        if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
            break

    t = 0
    while t < 10:
        if wstart == 0:
            curwjitter = int(np.random.rand() * jitter_range)
        elif wend == curShapeOrg[2]:
            curwjitter = -int(np.random.rand() * jitter_range)
        else:
            curwjitter = int(np.random.rand() * jitter_range) - jitter_range_half
        t += 1
        if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
            break

    if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
        cursplit[0][0] = curzjitter + zstart
        cursplit[0][1] = curzjitter + zend

    if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
        cursplit[1][0] = curhjitter + hstart
        cursplit[1][1] = curhjitter + hend

    if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
        cursplit[2][0] = curwjitter + wstart
        cursplit[2][1] = curwjitter + wend
    # print ("after ", cursplit)
    return cursplit


def augment(sample, label, coord=None, ifflip=True, ifswap=False, ifsmooth=False, ifjitter=False):
    """
    :param sample, the cropped sample input
    :param label, the corresponding sample ground-truth
    :param coord, the corresponding sample coordinates
    :param ifflip, flag for random flipping
    :param ifswap, flag for random swapping
    :param ifsmooth, flag for Gaussian smoothing on the CT image
    :param ifjitter, flag for intensity jittering on the CT image
    :return: augmented training samples
    """
    if ifswap:
        if sample.shape[0] == sample.shape[1] and sample.shape[0] == sample.shape[2]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, axisorder)
            label = np.transpose(label, axisorder)
            if coord is not None:
                coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))

    if ifflip:
        flipid = np.random.randint(2) * 2 - 1
        sample = np.ascontiguousarray(sample[:, :, ::flipid])
        label = np.ascontiguousarray(label[:, :, ::flipid])
        if coord is not None:
            coord = np.ascontiguousarray(coord[:, :, :, ::flipid])

    prob_aug = random.random()
    if ifjitter and prob_aug > 0.5:
        ADD_INT = (np.random.rand(sample.shape[0], sample.shape[1], sample.shape[2]) * 2 - 1) * 10
        ADD_INT = ADD_INT.astype('float')
        cury_roi = label * ADD_INT / 255.0
        sample += cury_roi
        sample[sample < 0] = 0
        sample[sample > 1] = 1

    prob_aug = random.random()
    if ifsmooth and prob_aug > 0.5:
        sigma = np.random.rand()
        if sigma > 0.5:
            sample = gaussian_filter(sample, sigma=1.0)

    return sample, label, coord

class OHEM_dataset(Dataset):
    def __init__(self,OHEM_list):
        self.OHEM_list = OHEM_list
    def __len__(self):
        return len(self.OHEM_list)
    def __getitem__(self, idx):
        return self.OHEM_list[idx]