import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAlpha(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))  # 初始化0，sigmoid后为0.5

    def forward(self, x):
        return torch.sigmoid(self.alpha)  # 保持输入输出维度一致

class FirstOctaveConv(nn.Module):
    """
    第一阶段八度卷积：将输入特征拆分为高频和低频分量（初始分离）
    核心作用：首次拆分频率特征，高频保留细节，低频保留全局结构
    输入：单尺度特征图 [B, in_channels, H, W]
    输出：
        X_h: 高频特征 [B, in_channels - int(alpha*in_channels), H, W]
        X_l: 低频特征 [B, int(alpha*in_channels), H//2, W//2]
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（与输入一致，仅拆分频率）
        kernel_size: 卷积核尺寸
        alpha: 低频通道占比（默认0.5）
        stride: 步幅（2时先下采样）
        padding/dilation/groups/bias: 卷积参数
    """

    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]  # 统一为单 kernel 尺寸（默认3）
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # 高频→低频下采样
        # 高频→低频卷积（通道数=alpha*in_channels）
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        # 高频→高频卷积（通道数=in_channels - alpha*in_channels）
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):  # x：[B, C, H, W]
        # 步幅为2时先整体下采样
        if self.stride == 2:
            x = self.h2g_pool(x)
        # 高频→低频：下采样+卷积
        X_h2l = self.h2g_pool(x)  # [B, C, H//2, W//2]
        # 高频特征增强
        X_h = self.h2h(x)  # [B, C - alpha*C, H, W]
        # 低频特征增强
        X_l = self.h2l(X_h2l)  # [B, alpha*C, H//2, W//2]
        return X_h, X_l


class OctaveConv(nn.Module):
    """
    中间阶段八度卷积：接收高低频特征，分别增强后输出新的高低频特征
    核心作用：跨频率交互+同频率增强，强化高低频关联
    输入：(X_h, X_l) 高频+低频特征对
    输出：
        X_h_new: 增强后高频 [B, out_channels - low_out, H', W']
        X_l_new: 增强后低频 [B, low_out, H'//2, W'//2]
    """

    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # 高频→低频下采样
        self.stride = stride

        # 固定通道分割比例（用于创建卷积层）
        self.low_in_channels = int(alpha * in_channels)
        self.high_in_channels = in_channels - self.low_in_channels
        self.low_out_channels = int(alpha * out_channels)
        self.high_out_channels = out_channels - self.low_out_channels

        # 四个频率交互卷积：覆盖同频/跨频增强
        # 低频→低频（同频增强）
        self.l2l = nn.Conv2d(self.low_in_channels, self.low_out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)
        # 低频→高频（跨频增强）
        self.l2h = nn.Conv2d(self.low_in_channels, self.high_out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)
        # 高频→低频（跨频增强）
        self.h2l = nn.Conv2d(self.high_in_channels, self.low_out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)
        # 高频→高频（同频增强）
        self.h2h = nn.Conv2d(self.high_in_channels, self.high_out_channels,
                             kernel_size, stride, padding, dilation, groups, bias)

        self.fuse_weight_h = AdaptiveAlpha(self.high_out_channels)
        self.fuse_weight_l = AdaptiveAlpha(self.low_out_channels)

    def forward(self, x):
        X_h, X_l = x  # 输入高低频特征对

        # 步幅为2时整体下采样
        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        # 四个频率交互路径
        X_l2l = self.l2l(X_l)  # 低频→低频
        X_h2l = self.h2l(self.h2g_pool(X_h))  # 高频→低频（需先下采样）
        X_l2h = self.l2h(X_l)  # 低频→高频
        X_h2h = self.h2h(X_h)  # 高频→高频

        # 对齐尺寸：上采样低频→高频分支
        X_l2h = F.interpolate(X_l2h, size=X_h2h.shape[2:], mode='bilinear', align_corners=False)

        # 融合生成新的高低频特征
        X_h_new = X_h2h + X_l2h  # 高频融合（跨频+同频）
        X_l_new = X_h2l + X_l2l  # 低频融合（跨频+同频）

        # 自适应加权（可选）：用学习到的权重调节高低频贡献
        # 将高低频拼接后计算注意力权重
        h_weight = self.fuse_weight_h(X_h_new)  # [B, high_out, 1, 1]
        l_weight = self.fuse_weight_l(X_l_new)  # [B, low_out, 1, 1]
        return X_h_new * h_weight, X_l_new * l_weight


class LastOctaveConv(nn.Module):
    """
    最后阶段八度卷积：将高低频特征融合为单尺度特征（最终聚合）
    核心作用：对齐高低频尺寸与通道，输出统一特征
    输入：(X_h, X_l) 高频+低频特征对
    输出：融合后单尺度特征 [B, out_channels, H, W]
    """

    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # 下采样（可选）
        # 低频→高频卷积（对齐通道）
        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        # 高频→高频卷积（对齐通道）
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x  # 输入高低频特征对
        # 步幅为2时整体下采样
        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        # 高频特征对齐通道
        X_h2h = self.h2h(X_h)  # [B, out_channels, H, W]
        # 低频特征：卷积+上采样（对齐高频尺寸和通道）
        X_l2h = self.l2h(X_l)  # [B, out_channels, H//2, W//2]
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        # 高低频融合输出
        X_h = X_h2h + X_l2h
        return X_h


class FAM(nn.Module):
    """
    频率自适应模块（FAM）：基于八度卷积的高低频分离-增强-融合模块
    核心创新：
        1. 三阶段频率处理：初始分离→多轮增强→最终融合，覆盖全频率特征；
        2. 跨频+同频交互：强化高低频关联，避免频率孤立；
        3. 自适应频率分配：通过alpha参数灵活调整高低频通道占比；
        4. 即插即用：输入输出均为单尺度特征，适配现有CNN架构。
    输入：单尺度特征图 [B, in_channels, H, W]
    输出：频率增强后单尺度特征 [B, out_channels, H, W]
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核尺寸（默认(3,3)）
    """

    def __init__(self, in_channels, kernel_size=(3, 3)):
        super(FAM, self).__init__()
        # 阶段1：初始频率分离（单尺度→高低频）
        out_channels = in_channels
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)
        # 阶段2：多轮频率增强（高低频→高低频，2轮同频+1轮跨频）
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)  # 同通道增强
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)  # 通道转换增强
        # 阶段3：最终频率融合（高低频→单尺度）
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)
        self.residual =  nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        # 阶段1：初始分离为高低频
        x_h, x_l = self.fir(x)

        # 阶段2：多轮增强（2轮mid1强化+1轮mid2通道转换）
        x_h_1, x_l_1 = self.mid1((x_h, x_l))
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1))
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2))

        # 阶段3：融合为单尺度特征
        x_ret = self.lst((x_h_5, x_l_5))
        return x_ret+residual


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 32, 32).to(device)

    model = FAM(64).to(device)
    y = model(x)

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)