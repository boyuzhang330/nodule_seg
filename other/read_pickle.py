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
    # data_path = r'D:\work\dataSet\Artery\crop_data'
    # data_path = r'D:\work\dataSet\nodule_data\crop_data'
    # data_path = r'/home/zhangboyu/dataset/airway/crop_data'
    data_path = r'/mnt/data2/boyu/nodule_data/scale123/'
    # ct_list = [os.path.join(data_path,i) for i in os.listdir(data_path) if not i.endswith('_label.nii.gz')]
    ct_list = [i.replace('_s1_label.nii.gz','') for i in os.listdir(data_path) if i.endswith('_s1_label.nii.gz')]
    ct_list.sort()
    # print(ct_list)

    train_name_list = ct_list[:1200]
    test_name_list = ct_list[1200:]

    train_list = []
    test_list = []
    for i in os.listdir(data_path):
        name = i.split('_s')[0]
        if 'label' in i:
            continue
        if name in train_name_list:
            train_list.append(os.path.join(data_path,i))
        else:
            test_list.append(os.path.join(data_path,i))
        # print(name)




    # train_list = ct_list[:1200]
    # val_list = ct_list[1200:]
    # test_list = ct_list[1200:]
    print(len(train_list))
    val_list = test_list
    dit= {'train':{'lidc':train_list},
          'val':{'lidc':val_list},
          'test':{'lidc':test_list}}
    file_name = 'split_nodule_stage1scale123.pickle'
    if os.path.exists(file_name):
        os.remove(file_name)
        print('remove')
    picklesave(dit,file_name)
    # print(dit)
