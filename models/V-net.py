import torch
import torch.nn as nn
import torch.nn.functional as F
from models.artery import ProjectExciteLayer

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.InstanceNorm3d(32)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        out = self.relu1(x)
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.InstanceNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        x = self.conv1(x)
        out = self.bn1(x)
        # split input in to 16 channels
        # x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 0)
        # print(x16.shape)
        # out = self.relu1(torch.add(out, x16))
        out = self.relu1(out)
        # print(out.shape)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.bn1(x)

        down = self.relu1(x)
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

        self.relu11 = ELUCons(elu, 35)
        self.conv11 = nn.Conv3d(35, 35, kernel_size=5, padding=2)
        self.bn11 = nn.InstanceNorm3d(35)
        self.relu12 = ELUCons(elu, 35)
        self.conv12 = nn.Conv3d(35, 32, kernel_size=5, padding=2)
        self.bn12 = nn.InstanceNorm3d(32)
    def forward(self, x, skipx,coordmap=None):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        if coordmap is not None:
            xcat = torch.cat((out, skipxdo,coordmap), 1) # 35
            out = self.relu11(xcat)
            out = self.conv11(out)
            out = self.bn11(out)
            out = self.relu2(torch.add(out, xcat))
            out = self.relu12(out)
            out = self.conv12(out)
            out = self.bn12(out)
        else:
            xcat = torch.cat((out, skipxdo), 1)
            out = self.ops(xcat)
            out = self.relu2(torch.add(out, xcat))


        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = nn.InstanceNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False,in_channels=1, out_channels=1,coord=True, Dmax=64, Hmax=64, Wmax=64):
        super(VNet, self).__init__()
        self._coord = coord
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

        self.pe1 = ProjectExciteLayer(16, Dmax, Hmax, Wmax)
        self.pe2 = ProjectExciteLayer(32, Dmax // 2, Hmax // 2, Wmax // 2)
        self.pe3 = ProjectExciteLayer(64, Dmax // 4, Hmax // 4, Wmax // 4)
        self.pe4 = ProjectExciteLayer(128, Dmax // 8, Hmax // 8, Wmax // 8)
        self.pe5 = ProjectExciteLayer(256, Dmax // 16, Hmax // 16, Wmax // 16)
        self.pe6 = ProjectExciteLayer(256, Dmax // 8, Hmax // 8, Wmax // 8)
        self.pe7 = ProjectExciteLayer(128, Dmax // 4, Hmax // 4, Wmax // 4)
        self.pe8 = ProjectExciteLayer(64, Dmax // 2, Hmax // 2, Wmax // 2)
        self.pe9 = ProjectExciteLayer(32, Dmax, Hmax, Wmax)
    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x,coordmap=None):
        out16 = self.in_tr(x)
        print('out16',out16.shape)
        out16, _ = self.pe1(out16)
        out32 = self.down_tr32(out16)
        print('out32',out32.shape)
        out32, _ = self.pe2(out32)
        out64 = self.down_tr64(out32)
        print('out64',out64.shape)
        out64, _ = self.pe3(out64)
        out128 = self.down_tr128(out64)
        print('out128',out128.shape)
        out128, _ = self.pe4(out128)

        out256 = self.down_tr256(out128)
        print('out256',out256.shape)
        out256, _ = self.pe5(out256)

        out = self.up_tr256(out256, out128)
        print('out',out.shape)
        out, _ = self.pe6(out)

        out = self.up_tr128(out, out64)
        print('out',out.shape)
        out, _ = self.pe7(out)

        out = self.up_tr64(out, out32)
        print('out',out.shape)
        out, _ = self.pe8(out)
        if (self._coord is True) and (coordmap is not None):
            out = self.up_tr32(out, out16,coordmap=coordmap)
        else:
            out = self.up_tr32(out, out16)

        print('out',out.shape)
        out, _ = self.pe9(out)



        out = self.out_tr(out)
        return out
if __name__ == '__main__':

    net = VNet()  # print(net)
    # print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
    x = torch.ones([2, 1, 64, 64, 64])
    coor = torch.ones([2, 3, 64, 64, 64])
    y = net(x,coor)
    print(y.shape)