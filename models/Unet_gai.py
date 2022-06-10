import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """
    Baseline model for nodule segmentation
    """

    def __init__(self, in_channels=1, out_channels=1):
        """
        :param in_channels: input channel numbers
        :param out_channels: output channel numbers
        """
        super(UNet3D, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), padding=0)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self._in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 64, 2, 2, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 32, 3, 1, 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 1, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))




        self.conv6 = nn.Conv3d(64, self._out_channels, 1, 1, 0)

    def forward(self, input):
        """
        :param input: shape = (batch_size, num_channels, D, H, W) \
        :param coordmap: shape = (batch_size, 2, D, H, W)
        :return: output segmentation tensor, attention mapping
        """
        conv1 = self.conv1(input)
        print(conv1.shape)
        conv2 = self.pooling(conv1)
        conv2 = self.conv2(conv2)
        print(conv2.shape)
        conv3 = self.pooling(conv2)
        conv3 = self.conv3(conv3)
        print(conv3.shape)

        conv4 = torch.cat([conv2, conv3], dim=1)
        conv4 = self.conv4(conv4)
        print(conv4.shape)

        conv5 = torch.cat([conv1,conv4],1)
        conv5 = self.conv5(conv5)
        print(conv5.shape)

        x = self.conv6(conv5)

        return x

    def crop(self,src, tar):
        z = src.shape[2] - tar.shape[2]
        y = src.shape[3] - tar.shape[3]
        x = src.shape[4] - tar.shape[4]
        src = src[:,:,int(z//2):int(src.shape[2]-z//2),
              int(y//2):int(src.shape[3]-y//2),
              int(x//2):int(src.shape[4]-x//2)]
        return src
if __name__ == '__main__':
    # 结节分割 输入dim:68

    net = UNet3D(in_channels=1, out_channels=2)
    # # print(net)
    # # print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
    dim = 64
    a = torch.randn([1, 1, dim, dim, dim])
    b = net(a)
    print(b.shape)



    a = torch.ones([1,1,28,28,28])
    b = torch.ones([1,1,20,20,20])
