import torch
import torch.nn as nn


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
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 128, 2, 2, 0),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 256, 2, 2, 0),
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(
            nn.Conv3d(256 + 256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 128, 2, 2, 0),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv3d(128 + 128, 64, 3, 1, 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 64, 2, 2, 0),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.Conv3d(64 + 64, 32, 3, 1, 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 32, 2, 2, 0),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.Conv3d(32 + 32, 32, 3, 1, 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 1, 1, 0),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True))

        # self.sigmoid = nn.Sigmoid()
        self.conv10 = nn.Conv3d(64, self._out_channels, 1, 1, 0)

    def forward(self, input):
        """
        :param input: shape = (batch_size, num_channels, D, H, W) \
        :param coordmap: shape = (batch_size, 2, D, H, W)
        :return: output segmentation tensor, attention mapping
        """
        conv1 = self.conv1(input)

        x = self.pooling(conv1)
        conv2 = self.conv2(x)

        x = self.pooling(conv2)
        conv3 = self.conv3(x)

        x = self.pooling(conv3)
        conv4 = self.conv4(x)

        # x = self.pooling(conv4)
        # conv5 = self.conv5(x)

        # x = torch.cat([conv4, conv5], dim=1)
        # conv6 = self.conv6(x)

        x = torch.cat([conv3, conv4], dim=1)
        conv7 = self.conv7(x)

        x = torch.cat([conv2, conv7], dim=1)
        conv8 = self.conv8(x)

        x = torch.cat([conv1, conv8], dim=1)
        conv9 = self.conv9(x)

        x = self.conv10(conv9)

        return x


if __name__ == '__main__':
    # 肺叶分割 输入dim:80

    net = UNet3D(in_channels=1, out_channels=2)
    # # print(net)
    # # print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
    dim = 80
    a = torch.randn([1, 1, 72, 80, 80])
    b = net(a)
    print(b.shape)
