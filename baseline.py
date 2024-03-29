import torch
import torch.nn as nn
import numpy as np
from models.Unet_5 import UNet3D as unet5
from models.Unet import UNet3D as unet
from models.airway import UNet3D
from models.vnet import VNet
from models.vnet_3 import VNet_3
from models.boundary_aware import UNet3D as ba_unet3d
from models.airway_dp import UNet3D as airway_dp_Unet


# Baseline

def get_model(conf, args=None):
    if conf['airway'] == 1:
        if conf['Vnet'] == 1:
            net = VNet(coord=True, Dmax=args.cubesize[0], Hmax=args.cubesize[1], Wmax=args.cubesize[2])
        elif conf['Vnet_3'] == 1:
            net = VNet_3(coord=True, Dmax=args.cubesize[0], Hmax=args.cubesize[1], Wmax=args.cubesize[2])
        elif conf['boundary_aware'] == 1:
            print('-------------boundary aware net---------------')
            net = ba_unet3d(in_channels=1, out_channels=args.out_channels, coord=True, Dmax=args.cubesize[0],
                            Hmax=args.cubesize[1], Wmax=args.cubesize[2])

        else:
            net = UNet3D(in_channels=1, out_channels=args.out_channels, coord=True, \
                         Dmax=args.cubesize[0], Hmax=args.cubesize[1], Wmax=args.cubesize[2])

    elif conf['suoxiao'] == 1:
        print('缩小')
        net = unet(in_channels=1, out_channels=args.out_channels)
    else:
        # net = unet5(in_channels=1, out_channels=args.out_channels)
        net = airway_dp_Unet(in_channels=1, out_channels=args.out_channels)

    # print(net)
    print('# of network parameters:', sum(param.numel() for param in net.parameters()))
    return net


if __name__ == '__main__':
    _, model = get_model()
