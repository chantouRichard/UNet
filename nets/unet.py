import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16

from nets.mda_parts import *

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # 修改点：参考 MDANet 的 DoubleConvWithAtt 逻辑
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.bn1    = nn.BatchNorm2d(out_size)
        self.relu   = nn.ReLU(inplace = True)
        self.att1   = MDA_CA_Block(out_size) # 引入曲线注意力

        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.bn2    = nn.BatchNorm2d(out_size)
        self.att2   = MDA_CA_Block(out_size) # 再次引入
        
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.att1(outputs) # 注意力应用

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)
        outputs = self.att2(outputs) # 注意力应用
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

        # 在 self.final 前面添加 side 层和输出层
        # 侧边输出通道数统一设为 16，参考 MDANet
        side_channels = 16

        # side4 对应 up4 (1/8尺度), side3 对应 up3 (1/4尺度)...
        self.side4 = Conv_Up(out_filters[3], side_channels, 8) 
        self.side3 = Conv_Up(out_filters[2], side_channels, 4)
        self.side2 = Conv_Up(out_filters[1], side_channels, 2)
        self.side1 = Conv_Up(out_filters[0], side_channels, 0) # 1/1尺度

        # 分别对应的评分层 (OutConv)
        self.score1 = OutConv(side_channels, num_classes)
        self.score2 = OutConv(side_channels, num_classes)
        self.score3 = OutConv(side_channels, num_classes)
        self.score4 = OutConv(side_channels, num_classes)
        self.score_final = OutConv(side_channels, num_classes) # 最终融合层
        
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

        # --- MDA 核心逻辑：侧边输出与融合 ---
        s4 = self.side4(up4)
        s3 = self.side3(up3)
        s2 = self.side2(up2)
        s1 = self.side1(up1)

        # 各尺度预测图 (用于计算多准则 Loss)
        score4 = self.score4(s4)
        score3 = self.score3(s3)
        score2 = self.score2(s2)
        score1 = self.score1(s1)

        # 最终融合预测
        fuse = s1 + s2 + s3 + s4
        score_final = self.score_final(fuse)

        # 返回 5 个输出，对应 MDANet 的训练需求
        return score1, score2, score3, score4, score_final

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
                    
    def set_mdanet_trainable(self, trainable=True):
        """专门控制 MDANet 的 5 个输出分支和 MDA 模块"""
        for name, param in self.named_parameters():
            # 锁定 side 输出层、融合层 (fuse) 以及 MDA 注意力模块
            if any(x in name.lower() for x in ['side', 'fuse', 'mda', 'transform']):
                param.requires_grad = trainable
                if trainable:
                    print(f"训练 MDANet 模块: {name}")

    def mdanet_freeze_strategy(self, stage):
        """
        根据阶段执行组合策略
        stage 1: 仅训练新模块 (CBAM + MDANet + Decoder)
        stage 2: 训练新模块 + Backbone 高层
        stage 3: 全解冻
        """
        # 先默认全部解冻，再根据阶段精细锁定
        self.unfreeze_backbone()
        
        if stage == 1:
            self.freeze_backbone() # 锁死 VGG
            self.set_cbam_trainable(True) # 练 CBAM
            self.set_mdanet_trainable(True) # 练 MDA
            print(">>> 策略：仅训练注意力与预测分支")
            
        elif stage == 2:
            self.set_backbone_partial_unfreeze() # 练 VGG 高层
            self.set_cbam_trainable(True)
            self.set_mdanet_trainable(True)
            print(">>> 策略：微调 VGG 高层 + 训练新模块")
            
        elif stage == 3:
            # 全部 True
            print(">>> 策略：全网络深度微调")