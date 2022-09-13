import os
from utils import load_itk_image,save_itk

input_path = r'/home/zhangboyu/dataset/nodule/crop_data2'
out_path = r'/home/zhangboyu/dataset/nodule/crop_data'
for i in os.listdir(input_path):
    print(i)
    arr,origin,spacing = load_itk_image(os.path.join(input_path,i))
    save_itk(arr,origin,spacing,os.path.join(out_path,i+'.gz'))
