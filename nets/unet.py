import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.lsnet import LSConv

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # 将原来的 nn.Conv2d 替换为 LSConv
        # conv1 的输入通道是 in_size，输出是 out_size
        self.conv1  = LSConv(dim = out_size) # 注意：LSConv内部通常in/out维相同，或需根据LSConv定义调整
        # 如果你的LSConv定义是LSConv(dim)，通常它是一个不改变通道数的算子
        # 针对标准的U-Net逻辑，我们先用一个1x1卷积调整通道，再接LSConv，或者直接替换
        
        # 更加稳健的替换方式（保持通道转换逻辑）：
        self.adjust_chan = nn.Conv2d(in_size, out_size, 1) # 先把拼接后的通道降维
        self.conv1  = LSConv(dim = out_size) 
        self.conv2  = LSConv(dim = out_size)
        
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.adjust_chan(outputs) # 调整通道数以适配 LSConv
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

    def set_cbam_trainable(self, trainable=True):
        """设置CBAM模块是否可训练"""
        for name, param in self.named_parameters():
            if 'cbam' in name.lower() or 'attention' in name.lower():
                param.requires_grad = trainable
                print(f"{'训练' if trainable else '冻结'}: {name}")

    def set_backbone_partial_unfreeze(self):
        """部分解冻VGG的高层（最后几层）"""
        if self.backbone == "vgg":
            # VGG有31层（0-30），解冻最后8层（23-30）
            for i, (name, param) in enumerate(self.vgg.named_parameters()):
                # 解冻features中后8层的参数
                if i >= 23:  # 最后8层
                    param.requires_grad = True
                    print(f"解冻高层: {name}")
                else:
                    param.requires_grad = False