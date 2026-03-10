import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
from .legnet import LFE_Module

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
        
        # --- 新增：为每个 Skip Connection 定义 LFE 增强模块 ---
        # 这里的 dim 对应 Encoder 输出的通道数
        # VGG: feat1(64), feat2(128), feat3(256), feat4(512)
        # ResNet50: feat1(64), feat2(256), feat3(512), feat4(1024)
        if backbone == 'vgg':
            feat_channels = [64, 128, 256, 512]
        else:
            feat_channels = [64, 256, 512, 1024]
        # 第一层用 Scharr 提取边缘，后续层用 Gaussian 去噪
        self.lfe1 = LFE_Module(feat_channels[0], stage=0, mlp_ratio=2, drop_path=0.1, 
                               act_layer=nn.ReLU, norm_layer=dict(type='BN'))
        self.lfe2 = LFE_Module(feat_channels[1], stage=1, mlp_ratio=2, drop_path=0.1, 
                               act_layer=nn.ReLU, norm_layer=dict(type='BN'))
        self.lfe3 = LFE_Module(feat_channels[2], stage=1, mlp_ratio=2, drop_path=0.1, 
                               act_layer=nn.ReLU, norm_layer=dict(type='BN'))
        self.lfe4 = LFE_Module(feat_channels[3], stage=1, mlp_ratio=2, drop_path=0.1, 
                               act_layer=nn.ReLU, norm_layer=dict(type='BN'))
        # --------------------------------------------------

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # --- 新增：在特征融合前进行增强 ---
        # feat5 是最底层(Bottleneck)，通常不需要处理，或者也可以加一个
        feat4_e = self.lfe4(feat4)
        feat3_e = self.lfe3(feat3)
        feat2_e = self.lfe2(feat2)
        feat1_e = self.lfe1(feat1)
        # -------------------------------

        # 使用增强后的特征进行上采样融合
        up4 = self.up_concat4(feat4_e, feat5)
        up3 = self.up_concat3(feat3_e, up4)
        up2 = self.up_concat2(feat2_e, up3)
        up1 = self.up_concat1(feat1_e, up2)

        if self.up_conv is not None:
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