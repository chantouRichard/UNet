import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16

from nets.ASPP import ASPP

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
        
        self.aspp = ASPP(dim_in=512, dim_out=out_filters[3], rate=1, bn_mom=0.1)
        
        self.has_aspp = True  # 标记是否使用了ASPP

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # ---------- 在bottleneck处应用ASPP ----------
        # 对最深层的特征feat5进行多尺度上下文提取
        feat5_aspp = self.aspp(feat5)
        
        up4 = self.up_concat4(feat4, feat5_aspp)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        """冻结backbone，但保持ASPP和decoder可训练"""
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 确保ASPP是训练状态（如果存在）
        if hasattr(self, 'aspp'):
            for param in self.aspp.parameters():
                param.requires_grad = True
            print("ASPP保持训练状态")

    def unfreeze_backbone(self):
        """解冻所有参数"""
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
        
        # 确保ASPP是训练状态
        if hasattr(self, 'aspp'):
            for param in self.aspp.parameters():
                param.requires_grad = True

    def set_cbam_trainable(self, trainable=True):
        """设置CBAM模块是否可训练（如果存在的话）"""
        found = False
        for name, param in self.named_parameters():
            if 'cbam' in name.lower() or 'attention' in name.lower():
                param.requires_grad = trainable
                print(f"{'训练' if trainable else '冻结'}: {name}")
                found = True
        
        # 可以顺便打印ASPP状态
        if hasattr(self, 'aspp'):
            print(f"ASPP模块默认是可训练的")
        
        if not found:
            print("未找到CBAM模块")

    def set_backbone_partial_unfreeze(self):
        """部分解冻VGG的高层（最后几层）"""
        if self.backbone == "vgg":
            # VGG有31层（0-30），解冻最后8层（23-30）
            unfrozen_count = 0
            for i, (name, param) in enumerate(self.vgg.named_parameters()):
                # 解冻features中后8层的参数
                if i >= 23:  # 最后8层
                    param.requires_grad = True
                    print(f"解冻高层: {name}")
                    unfrozen_count += 1
                else:
                    param.requires_grad = False
            print(f"已解冻VGG最后 {unfrozen_count} 个参数组")
            
        elif self.backbone == "resnet50":
            # 对于ResNet，可以解冻layer4和layer3
            print("解冻ResNet的layer3和layer4")
            for name, param in self.resnet.named_parameters():
                if 'layer4' in name or 'layer3' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # ASPP保持可训练
        if hasattr(self, 'aspp'):
            for param in self.aspp.parameters():
                param.requires_grad = True
            print("ASPP保持训练状态")