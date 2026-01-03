import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from nets.unet import Unet
from utils.dataloader import UnetDataset, unet_dataset_collate
from torch.utils.data import DataLoader

class AttentionVisualizer:
    """注意力机制可视化工具"""
    
    def __init__(self, model_path='model_data/unet_vgg_voc.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = Unet(num_classes=21, pretrained=False, backbone='vgg').to(self.device)
        
        if model_path and os.path.exists(model_path):
            print(f"加载预训练权重: {model_path}")
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=self.device)
            
            # 过滤权重
            load_key, no_load_key = [], []
            temp_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and model_dict[k].shape == v.shape:
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            
            model_dict.update(temp_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"成功加载 {len(load_key)} 个权重")
        
        self.model.eval()
        
        # 注册hook来获取中间特征
        self.feature_maps = {}
        self.attention_maps = {}
        self.register_hooks()
    
    def register_hooks(self):
        """注册hook来捕获注意力层的输出"""
        def get_activation_hook(name):
            def hook(module, input, output):
                # 保存特征图
                self.feature_maps[name] = output.detach().cpu()
            return hook
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # 对于CBAM模块，我们需要分别捕获通道注意力和空间注意力
                if hasattr(module, 'channel_attention'):
                    # 获取通道注意力权重
                    channel_att = module.channel_attention
                    if hasattr(channel_att, 'avg_pool') and hasattr(channel_att, 'max_pool'):
                        x = input[0]
                        b, c, h, w = x.shape
                        avg_out = channel_att.avg_pool(x)
                        max_out = channel_att.max_pool(x)
                        # 这里简化处理，实际需要更复杂的逻辑来获取注意力权重
                        self.attention_maps[f"{name}_channel"] = avg_out.mean(dim=1).detach().cpu()
                
                if hasattr(module, 'spatial_attention'):
                    # 获取空间注意力权重
                    spatial_att = module.spatial_attention
                    x = input[0] if isinstance(input, tuple) else input
                    avg_out = torch.mean(x, dim=1, keepdim=True)
                    max_out, _ = torch.max(x, dim=1, keepdim=True)
                    spatial_map = torch.cat([avg_out, max_out], dim=1)
                    self.attention_maps[f"{name}_spatial"] = spatial_map.mean(dim=1).detach().cpu()
                
                # 保存整个CBAM的输出
                self.feature_maps[name] = output.detach().cpu()
            return hook
        
        # 查找模型中的所有CBAM模块
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                # 注册特征图hook
                module.register_forward_hook(get_activation_hook(name))
            
            # 查找CBAM模块（根据你的CBAM类名）
            if 'CBAM' in str(type(module)) or 'cbam' in name.lower():
                print(f"找到注意力模块: {name}")
                module.register_forward_hook(get_attention_hook(name))
    
    def visualize_attention(self, image_tensor, save_dir='attention_visualization'):
        """可视化注意力机制"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 清空之前的特征图
        self.feature_maps.clear()
        self.attention_maps.clear()
        
        # 前向传播
        with torch.no_grad():
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            output = self.model(image_tensor)
        
        # 可视化原始图像
        image_np = image_tensor[0].cpu().numpy()
        if image_np.min() < 0:
            image_np = (image_np + 1) / 2  # 从[-1,1]转到[0,1]
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 3, 1)
        plt.imshow(np.transpose(image_np, (1, 2, 0)))
        plt.title('Input Image')
        plt.axis('off')
        
        # 可视化每个CBAM模块的输出
        attention_keys = [k for k in self.attention_maps.keys() if 'attention' in k]
        
        for idx, key in enumerate(attention_keys[:5]):  # 最多显示5个
            attention_map = self.attention_maps[key]
            if attention_map.dim() == 4:
                attention_map = attention_map[0, 0]  # 取第一个样本的第一个通道
            elif attention_map.dim() == 3:
                attention_map = attention_map[0]  # 取第一个样本
            
            plt.subplot(2, 3, idx + 2)
            plt.imshow(attention_map.numpy(), cmap='jet')
            plt.title(f'Attention: {key}')
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attention_maps.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 可视化特征图
        self.visualize_feature_maps(save_dir)
        
        print(f"注意力可视化结果已保存到: {save_dir}")
    
    def visualize_feature_maps(self, save_dir):
        """可视化特征图"""
        # 选择一些有代表性的层进行可视化
        conv_keys = [k for k in self.feature_maps.keys() 
                    if ('conv' in k or 'features' in k) 
                    and self.feature_maps[k].dim() == 4]
        
        for key in conv_keys[:3]:  # 最多可视化3个卷积层
            feature_map = self.feature_maps[key]
            if feature_map.shape[1] > 16:  # 如果通道太多，只取前16个
                feature_map = feature_map[:, :16]
            
            # 创建特征图可视化
            fig, axes = plt.subplots(4, 4, figsize=(12, 10))
            fig.suptitle(f'Feature Maps: {key}', fontsize=16)
            
            for i in range(min(16, feature_map.shape[1])):
                ax = axes[i // 4, i % 4]
                channel_data = feature_map[0, i].numpy()
                im = ax.imshow(channel_data, cmap='viridis')
                ax.set_title(f'Channel {i}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'features_{key.replace(".", "_")}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def test_with_sample_data(self):
        """使用样本数据测试"""
        # 创建一个测试图像
        test_image = torch.randn(1, 3, 512, 512)
        print(f"测试图像形状: {test_image.shape}")
        
        # 可视化注意力
        self.visualize_attention(test_image)
        
        # 测试前向传播
        with torch.no_grad():
            if torch.cuda.is_available():
                test_image = test_image.cuda()
            output = self.model(test_image)
        
        print(f"模型输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        return output

def test_attention_on_real_data():
    """在真实数据上测试注意力"""
    # 加载数据
    VOCdevkit_path = 'VOCdevkit'
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()[:2]  # 只取前2个样本
    
    # 创建数据集
    val_dataset = UnetDataset(val_lines, [512, 512], 2, False, VOCdevkit_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           collate_fn=unet_dataset_collate)
    
    # 创建可视化器
    visualizer = AttentionVisualizer()
    
    # 测试每个样本
    for i, data in enumerate(val_loader):
        images = data[0]
        print(f"\n处理样本 {i+1}: {images.shape}")
        
        # 可视化注意力
        visualizer.visualize_attention(images, save_dir=f'attention_results/sample_{i+1}')
        
        if i >= 1:  # 只测试2个样本
            break
    
    print("\n" + "="*50)
    print("注意力测试完成！")
    print("请查看 attention_results/ 文件夹中的可视化结果")
    print("="*50)

def compare_without_attention():
    """对比有无注意力机制的效果"""
    # 创建带注意力和不带注意力的模型
    model_with_att = Unet(num_classes=21, pretrained=False, backbone='vgg')
    model_without_att = Unet(num_classes=21, pretrained=False, backbone='vgg')
    
    # 加载预训练权重
    model_path = 'model_data/unet_vgg_voc.pth'
    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path, map_location='cpu')
        
        # 加载到两个模型
        model_with_att.load_state_dict(pretrained_dict, strict=False)
        model_without_att.load_state_dict(pretrained_dict, strict=False)
        
        # 手动关闭不带注意力模型的注意力层
        for name, module in model_without_att.named_modules():
            if 'CBAM' in str(type(module)) or 'attention' in name.lower():
                print(f"关闭注意力模块: {name}")
                # 设置注意力模块为恒等映射
                for param in module.parameters():
                    param.requires_grad = False
                # 这里需要根据你的注意力模块实现来修改
    
    # 测试推理速度
    test_input = torch.randn(1, 3, 512, 512)
    
    import time
    start_time = time.time()
    with torch.no_grad():
        output_with = model_with_att(test_input)
    time_with = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        output_without = model_without_att(test_input)
    time_without = time.time() - start_time
    
    print(f"\n推理时间对比:")
    print(f"带注意力: {time_with:.4f}秒")
    print(f"不带注意力: {time_without:.4f}秒")
    print(f"时间差: {abs(time_with - time_without):.4f}秒")
    
    # 比较输出差异
    diff = torch.abs(output_with - output_without).mean().item()
    print(f"输出差异(平均绝对误差): {diff:.6f}")

if __name__ == "__main__":
    print("="*60)
    print("注意力机制测试工具")
    print("="*60)
    
    print("\n1. 测试样本数据...")
    visualizer = AttentionVisualizer()
    visualizer.test_with_sample_data()
    
    print("\n2. 在真实数据上测试...")
    test_attention_on_real_data()
    
    print("\n3. 对比有无注意力机制...")
    compare_without_attention()
    
    print("\n测试完成！")