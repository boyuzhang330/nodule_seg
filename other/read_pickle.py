import pickle

def load_pickle(filename='split_dataset.pickle'):
    """
    :param filename: pickle name
    :return: dictionary or list
    """
    with open(filename, 'rb') as handle:
        ids = pickle.load(handle)
    return ids

def picklesave(obj,file):
    ff = open(file,'wb')
    pickle.dump(obj,ff)
    ff.close()

def pickleload(file):
    ff = open(file,'rb')
    obj = pickle.load(ff)
    # ff.close()
    return obj

if __name__ == '__main__':
    import os
    data_path1 = r'D:\work\dataSet\Artery\crop_data\\'
    # data_path2 = r'D:\work\dataSet\nodule_data\crop_data'
    data_path2 = r'/home/zhangboyu/dataset/Artery/crop_data/'

    # ct_list = [data_path2+i for i in os.listdir(data_path1) if not i.endswith('_label.nii.gz')]
    ct_list = [data_path1+i for i in os.listdir(data_path1) if not i.endswith('_label.nii.gz')]
    print(len(ct_list))

    # print(ct_list)
    # text = pickleload('split_dataset.pickle')
    # print(text)
    num = 90
    train_list = ct_list[:1]
    val_list = ct_list[:1]
    # val_list = ct_list[320:]
    test_list = ct_list[:1]
    print(len(train_list))
    dit= {'train':{'lidc':train_list},
          'val':{'lidc':val_list},
          'test':{'lidc':test_list}}
    file_name = 'split_artery.pickle'
    if os.path.exists(file_name):
        os.remove(file_name)
        print('remove')
    picklesave(dit,file_name)
    print(dit)
