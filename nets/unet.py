import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

import torch.nn.functional as F
class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg', deep_supervision = True):
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
        self.deep_supervision = deep_supervision

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

        self.ds_head4 = nn.Conv2d(out_filters[3], num_classes, kernel_size=1)
        self.ds_head3 = nn.Conv2d(out_filters[2], num_classes, kernel_size=1)
        self.ds_head2 = nn.Conv2d(out_filters[1], num_classes, kernel_size=1)
        
        # 如果你想引入 Refinement Module，可以在这里初始化
        # self.rm4 = Refinement_Module(out_filters[3]) 
        
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        # 1. Encoder 提取特征
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # 2. Decoder 解码及深度监督采样
        # up4: 1/16 尺度
        up4 = self.up_concat4(feat4, feat5)
        out4 = F.interpolate(self.ds_head4(up4), size=inputs.size()[2:], mode='bilinear', align_corners=False)

        # up3: 1/8 尺度
        up3 = self.up_concat3(feat3, up4)
        out3 = F.interpolate(self.ds_head3(up3), size=inputs.size()[2:], mode='bilinear', align_corners=False)

        # up2: 1/4 尺度
        up2 = self.up_concat2(feat2, up3)
        out2 = F.interpolate(self.ds_head2(up2), size=inputs.size()[2:], mode='bilinear', align_corners=False)

        # up1: 1/2 尺度 (或原图尺度，取决于你的 backbone 设计)
        up1 = self.up_concat1(feat1, up2)
        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        final_out = self.final(up1)
        if final_out.size()[2:] != inputs.size()[2:]:
            final_out = F.interpolate(final_out, size=inputs.size()[2:], mode='bilinear', align_corners=False)

        # 3. 根据模式返回输出
        if self.deep_supervision and self.training:
            # 训练阶段返回所有输出，用于计算总 Loss (Loss = L_final + α1*L4 + α2*L3 ...)
            return [out4, out3, out2, final_out]
        else:
            # 推理阶段只返回最终结果
            return final_out

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