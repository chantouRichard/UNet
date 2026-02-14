import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class VGGWithCBAM(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGGWithCBAM, self).__init__()
        self.features = features
        
        #-----------------------------#
        # 多通道输入Hession矩阵
        #-----------------------------#
        # 修改第一层输入通道数为4
        old_conv = self.features[0]
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        # 复制原来的3通道权重
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data

        # 第4通道用平均值初始化（关键）
        new_conv.weight.data[:, 3:4, :, :] = \
            old_conv.weight.data.mean(dim=1, keepdim=True)

        # bias复制
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        # 替换
        self.features[0] = new_conv
        
        # 在5个下采样阶段后添加CBAM注意力
        self.cbam1 = CBAM(channel=64)    # 对应feat1的输出通道
        self.cbam2 = CBAM(channel=128)   # 对应feat2的输出通道  
        self.cbam3 = CBAM(channel=256)   # 对应feat3的输出通道
        self.cbam4 = CBAM(channel=512)   # 对应feat4的输出通道
        self.cbam5 = CBAM(channel=512)   # 对应feat5的输出通道
        
        # VGG的分类器部分
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        # 第1阶段：前4层 → CBAM
        feat1 = self.features[:4](x)
        feat1 = self.cbam1(feat1)
        
        # 第2阶段：4-9层 → CBAM
        feat2 = self.features[4:9](feat1)
        feat2 = self.cbam2(feat2)
        
        # 第3阶段：9-16层 → CBAM
        feat3 = self.features[9:16](feat2)
        feat3 = self.cbam3(feat3)
        
        # 第4阶段：16-23层 → CBAM
        feat4 = self.features[16:23](feat3)
        feat4 = self.cbam4(feat4)
        
        # 第5阶段：23层到倒数第2层 → CBAM
        feat5 = self.features[23:-1](feat4)
        feat5 = self.cbam5(feat5)
        
        # 如果需要分类，继续执行
        # x = self.avgpool(feat5)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x
        
        # 对于UNet编码器，返回多尺度特征
        return [feat1, feat2, feat3, feat4, feat5]

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[  :4 ](x)
        feat2 = self.features[4 :9 ](feat1)
        feat3 = self.features[9 :16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels = 3, **kwargs):
    # model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    model = VGGWithCBAM(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    
    del model.avgpool
    del model.classifier
    return model
