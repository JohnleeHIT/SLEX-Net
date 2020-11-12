import torch
import torch.nn.functional as F
import torch.nn as nn


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, track_running_stats = True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats = track_running_stats),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats = track_running_stats),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, drop_rate=0.0, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, drop_rate)
        self.down2 = Down(128, 256, drop_rate)
        self.down3 = Down(256, 512, drop_rate)
        self.down4 = Down(512, 512, drop_rate)
        self.up1 = Up(1024, 256, drop_rate, bilinear)
        self.up2 = Up(512, 128, drop_rate, bilinear)
        self.up3 = Up(256, 64, drop_rate, bilinear)
        self.up4 = Up(128, 64, drop_rate, bilinear)
        self.outc = OutConv(64, n_classes)
        initialize_weights(self)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # softmax_out = F.softmax(logits, dim=1)
        return logits


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.drop_rate>0:
            x = F.dropout(x, p=self.drop_rate, training=True)
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop_rate, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        self.drop_rate = drop_rate

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        if self.drop_rate>0:
            x = F.dropout(x, p=self.drop_rate, training=True)
        return self.conv(x)


class HybridUNet_single_out_consistency(nn.Module):
    def __init__(self, backbone_channel, backbone_class, drop_rate, SEM_channels, SEM_class):
        # define the hybrid network containing SEM
        super(HybridUNet_single_out_consistency, self).__init__()
        # test dropout
        self.backbone = Unet(n_channels=backbone_channel, drop_rate=drop_rate, n_classes=backbone_class)
        # self.SMM---a.k.a. SEM
        self.SMM = Unet(n_channels=SEM_channels, n_classes=SEM_class, drop_rate=drop_rate)

    def forward(self, input, lamda=0.5):
        # Here input is 3 adjacent slices
        # three backbone network
        input0 = torch.unsqueeze(input[:, 0, ...], 1)
        input1 = torch.unsqueeze(input[:, 1, ...], 1)
        input2 = torch.unsqueeze(input[:, 2, ...], 1)

        self.out0 = self.backbone(input0)
        self.out1 = self.backbone(input1)
        self.out2 = self.backbone(input2)

        # Choose the prediction of the foreground and other two slices as the Input of SEM module
        input_sem21 = torch.cat((input2, input1,torch.unsqueeze(self.out2[:,1, ...], 1)) ,dim=1)
        self.out1_sem_2 = self.SMM(input_sem21)

        input_sem01 = torch.cat((input0, input1, torch.unsqueeze(self.out0[:,1, ...], 1)), 1)
        self.out1_sem_0 = self.SMM(input_sem01)

        # consistency error weighting
        self.consistency_error = torch.exp(-(self.out1_sem_0-self.out1_sem_2)**2)*lamda
        self.hybrid = self.out1*(1-self.consistency_error*2)+self.out1_sem_2*self.consistency_error\
                      +self.out1_sem_0*self.consistency_error
        # self.hybrid = self.out1_sem_2 * lamda + lamda * self.out1_sem_0 + self.out1*(1-2*lamda)
        # output
        self.final_out = self.hybrid
        self.intermediate = torch.cat((self.out1_sem_2, self.out1_sem_0, self.out1),1)
        #  add final softmax output
        self.softmax_out = torch.cat((F.softmax(self.out1_sem_2, dim=1),
                                     F.softmax(self.out1_sem_0, dim=1),
                                     F.softmax(self.out1, dim=1),
                                     F.softmax(self.final_out, dim=1)), 1)

        return self


class HybridUNet_single_out_consistency2(nn.Module):
    def __init__(self, backbone_channel, backbone_class, drop_rate, SEM_channels, SEM_class):
        # define the hybrid network contaning SEM
        super(HybridUNet_single_out_consistency2, self).__init__()
        # test dropout
        self.backbone = Unet(n_channels=backbone_channel, drop_rate=drop_rate, n_classes=backbone_class)
        self.SEM = Unet(n_channels=SEM_channels, n_classes=SEM_class, drop_rate=drop_rate)

    def forward(self, input, lamda=0.5):
        # Here input is 3 adjacent slices
        # three backbone network
        input0 = torch.unsqueeze(input[:, 0, ...], 1)
        input1 = torch.unsqueeze(input[:, 1, ...], 1)
        input2 = torch.unsqueeze(input[:, 2, ...], 1)

        self.out0 = self.backbone(input0)
        self.out1 = self.backbone(input1)
        self.out2 = self.backbone(input2)

        # Choose the prediction of the foreground and other two slices as the Input of SEM module
        input_sem21 = torch.cat((input2, input1,torch.unsqueeze(self.out2[:,1, ...], 1)) ,dim=1)
        self.out1_sem_2 = self.SEM(input_sem21)

        input_sem01 = torch.cat((input0, input1, torch.unsqueeze(self.out0[:,1, ...], 1)), 1)
        self.out1_sem_0 = self.SEM(input_sem01)

        # consistency error weighting
        self.consistency_error = torch.exp(-(self.out1_sem_0-self.out1_sem_2)**2)*lamda
        self.hybrid = self.out1*(1-self.consistency_error*2)+self.out1_sem_2*self.consistency_error\
                      +self.out1_sem_0*self.consistency_error
        # self.hybrid = self.out1_sem_2 * lamda + lamda * self.out1_sem_0 + self.out1*(1-2*lamda)
        # output
        self.final_out = self.hybrid
        self.intermediate = torch.cat((self.out1_sem_2, self.out1_sem_0, self.out1),1)
        #  add final softmax output
        self.softmax_out = torch.cat((F.softmax(self.out1_sem_2, dim=1),
                                     F.softmax(self.out1_sem_0, dim=1),
                                     F.softmax(self.out1, dim=1),
                                     F.softmax(self.final_out, dim=1)), 1)

        return self


class HybridUNet_single_out_multi_slices(nn.Module):
    def __init__(self, backbone_channel, backbone_class, drop_rate, SEM_channels, SEM_class):
        # define the hybrid network contaning SEM
        super(HybridUNet_single_out_multi_slices, self).__init__()
        # test dropout
        self.backbone = Unet(n_channels=backbone_channel, drop_rate=drop_rate, n_classes=backbone_class)
        self.SEM = Unet(n_channels=SEM_channels, n_classes=SEM_class, drop_rate=drop_rate)

    def forward(self, input, num_slices, lamda_list):
        out_backbone_list = []
        input_list = []
        output_sem_list = []
        # Here input is multiple adjacent slices
        # multiple backbone network
        for i in range(num_slices):
            input_slice = torch.unsqueeze(input[:,i, ...], 1)
            out_backbone = self.backbone(input_slice)
            out_backbone_list.append(out_backbone)
            input_list.append(input_slice)

        # At first we should determine the target slice, start from zero
        target_slice = num_slices//2
        for j in range(num_slices):
            input_sem = torch.cat((input_list[j],input_list[target_slice],
                                   torch.unsqueeze(out_backbone_list[j][:,1,...],1)),dim=1)
            output_sem = self.SEM(input_sem)
            output_sem_list.append(output_sem)

        # first calculate the weighted target result
        self.final_out = out_backbone_list[target_slice]*lamda_list[target_slice]
        for k in range(num_slices):
            if k == target_slice:
                continue
            else:
                self.final_out += output_sem_list[k]*lamda_list[k]

        # replace the output_sem at target slice with backbone output for that slice
        output_sem_list[target_slice] = out_backbone_list[target_slice]

        self.intermediate = torch.cat(output_sem_list, 1)
        softmax_list = [F.softmax(output_sem_list[n], dim=1) for n in range(num_slices)]
        self.softmax_out = torch.cat(softmax_list, 1)

        return self


if __name__ == "__main__":
    import numpy as np