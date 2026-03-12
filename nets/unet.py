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
                    
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStageUnet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(TwoStageUnet, self).__init__()
        
        # 1. 共享 Encoder (特征提取器)
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]
        self.backbone = backbone

        # -----------------------------------------------------------
        # 2. Fine Decoder (保持原名，以便加载原有权重)
        # -----------------------------------------------------------
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        # -----------------------------------------------------------
        # 3. Coarse Decoder (新增，前缀加 coarse_)
        # -----------------------------------------------------------
        self.coarse_up4 = unetUp(in_filters[3], out_filters[3])
        self.coarse_up3 = unetUp(in_filters[2], out_filters[2])
        self.coarse_up2 = unetUp(in_filters[1], out_filters[1])
        self.coarse_up1 = unetUp(in_filters[0], out_filters[0])
        self.coarse_final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, x_full, x_patch=None):
        """
        x_full: 降采样后的低分辨率全图 (用于 Coarse)
        x_patch: 原始分辨率的局部 Patch (用于 Fine)
        """
        
        # --- STAGE 1: Coarse Segmentation (Global) ---
        if self.backbone == "vgg":
            c_feats = self.vgg(x_full)
        else:
            c_feats = self.resnet(x_full)
        
        c_up4 = self.coarse_up4(c_feats[3], c_feats[4])
        c_up3 = self.coarse_up3(c_feats[2], c_up4)
        c_up2 = self.coarse_up2(c_feats[1], c_up3)
        c_up1 = self.coarse_up1(c_feats[0], c_up2)
        coarse_out = self.coarse_final(c_up1)

        # 如果没有输入 patch（例如在单独测试全图时），直接返回粗略结果
        if x_patch is None:
            return coarse_out

        # --- STAGE 2: Fine Segmentation (Local) ---
        if self.backbone == "vgg":
            f_feats = self.vgg(x_patch)
        else:
            f_feats = self.resnet(x_patch)

        f_up4 = self.up_concat4(f_feats[3], f_feats[4])
        f_up3 = self.up_concat3(f_feats[2], f_up4)
        f_up2 = self.up_concat2(f_feats[1], f_up3)
        f_up1 = self.up_concat1(f_feats[0], f_up2)
        fine_out = self.final(f_up1)

        return coarse_out, fine_out

    def freeze_all_but_coarse(self):
        """阶段 1：冻结所有，只留下 Coarse 分支和 CBAM 可训练"""
        # 1. 冻结 Backbone
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 2. 冻结 Fine Decoder (因为已有旧权重)
        for name, param in self.named_parameters():
            if 'up_concat' in name or 'final' in name:
                if 'coarse' not in name: # 排除 coarse 分支
                    param.requires_grad = False
        
        # 3. 确保 Coarse 分支和 CBAM 是开启的
        for name, param in self.named_parameters():
            if 'coarse' in name or 'cbam' in name or 'attention' in name:
                param.requires_grad = True

    def set_backbone_partial_unfreeze(self):
        """阶段 2：解冻 Backbone 高层，并解冻 Fine 分支"""
        # 1. 解冻 Fine 分支
        for name, param in self.named_parameters():
            if 'up_concat' in name or 'final' in name:
                param.requires_grad = True
        
        # 2. 部分解冻 Backbone
        if self.backbone == "vgg":
            # VGG 最后几层通常是 index 23 以后
            for i, (name, param) in enumerate(self.vgg.named_parameters()):
                if i >= 23: 
                    param.requires_grad = True
                    print(f"解冻 Backbone 高层: {name}")
        elif self.backbone == "resnet50":
            # ResNet50 解冻 Layer4
            for name, param in self.resnet.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True
                    print(f"解冻 Backbone Layer4: {name}")

    def unfreeze_all(self):
        """阶段 3：全解冻"""
        for param in self.parameters():
            param.requires_grad = True
        print("全模型已解冻")

    # 原有的 set_cbam_trainable 可以保留作为辅助工具
    def set_cbam_trainable(self, trainable=True):
        for name, param in self.named_parameters():
            if 'cbam' in name.lower() or 'attention' in name.lower():
                param.requires_grad = trainable