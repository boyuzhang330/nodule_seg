import torch
import torch.nn as nn
import numpy as np
from models.Unet_5 import UNet3D as unet5
from models.Unet import UNet3D as unet
from models.artery import UNet3D

# Baseline

def get_model(conf,args=None):
    if conf['artery']==1:
        net = UNet3D(in_channels=1, out_channels=args.out_channels, coord=True, \
                Dmax=args.cubesize[0], Hmax=args.cubesize[1], Wmax=args.cubesize[2])

    elif conf['suoxiao']==1:
        print('缩小')
        net = unet(in_channels=1, out_channels=args.out_channels)
    else:
        net = unet5(in_channels=1, out_channels=args.out_channels)

    # print(net)
    print('# of network parameters:', sum(param.numel() for param in net.parameters()))
    return net


if __name__ == '__main__':
    _, model = get_model()
