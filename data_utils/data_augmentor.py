import cv2
import numpy as np
import os
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tqdm

class BridgeDataAugmentor:
    """桥梁数据增强器，同时增强图像和mask"""
    
    def __init__(self, output_dir=None):
        """
        初始化增强器
        
        参数:
            output_dir: 输出目录，默认为'augmented_data'
        """
        self.output_dir = output_dir or 'augmented_data'
        
        # 创建输出目录结构
        self.img_output_dir = os.path.join(self.output_dir, 'images')
        self.mask_output_dir = os.path.join(self.output_dir, 'masks')
        os.makedirs(self.img_output_dir, exist_ok=True)
        os.makedirs(self.mask_output_dir, exist_ok=True)
        
        # 定义桥梁图像特定的增强管道
        # 注意：mask需要和图像同步增强，所以使用相同的变换
        self.transform_pipelines = self._create_transform_pipelines()
        
        # 支持的图像格式
        self.img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    def _create_transform_pipelines(self):
        """创建多个增强管道，模拟不同的现实条件"""
        
        pipelines = []
        
        # 1. 基础几何变换（小幅度）
        pipelines.append(A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # 桥梁垂直翻转要谨慎
            A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),  # 小角度旋转
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=5, 
                              p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
        ], additional_targets={'mask': 'image'}))
        
        # 2. 颜色/光照变换
        pipelines.append(A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ], additional_targets={'mask': 'image'}))
        
        # 3. 天气/环境模拟
        pipelines.append(A.Compose([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.3),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, 
                         drop_width=1, drop_color=(200, 200, 200), p=0.2),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, 
                          shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.2),
        ], additional_targets={'mask': 'image'}))
        
        # 4. 传感器噪声模拟
        pipelines.append(A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.4),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.3),
        ], additional_targets={'mask': 'image'}))
        
        # 5. 模糊/运动模糊
        pipelines.append(A.Compose([
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
        ], additional_targets={'mask': 'image'}))
        
        # 6. 组合变换（中等强度）
        pipelines.append(A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=3, p=0.4),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        ], additional_targets={'mask': 'image'}))
        
        # 7. 组合变换（高强度）
        pipelines.append(A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=8, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.GaussNoise(var_limit=(20.0, 70.0), p=0.4),
        ], additional_targets={'mask': 'image'}))
        
        # 8. 透视变换（模拟不同拍摄角度）
        pipelines.append(A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=0.5, fit_output=False),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        ], additional_targets={'mask': 'image'}))
        
        return pipelines
    
    def find_image_mask_pairs(self, images_dir, masks_dir):
        """
        找出图像和mask的对应关系
        
        参数:
            images_dir: 原始图像目录
            masks_dir: mask图像目录
            
        返回:
            list of tuples: [(image_path, mask_path), ...]
        """
        pairs = []
        
        # 获取所有图像文件
        img_files = []
        for ext in self.img_extensions:
            img_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
        
        # 为每个图像寻找对应的mask
        for img_file in img_files:
            img_name_without_ext = os.path.splitext(img_file)[0]
            
            # 尝试不同的mask文件名模式
            mask_found = False
            for ext in self.mask_extensions:
                # 模式1: 同名文件
                mask_path = os.path.join(masks_dir, f"{img_name_without_ext}{ext}")
                if os.path.exists(mask_path):
                    pairs.append((os.path.join(images_dir, img_file), mask_path))
                    mask_found = True
                    break
                
                # 模式2: 带后缀（如_image,_mask）
                for suffix in ['_mask', '_label', '_seg', '_gt']:
                    mask_path = os.path.join(masks_dir, f"{img_name_without_ext}{suffix}{ext}")
                    if os.path.exists(mask_path):
                        pairs.append((os.path.join(images_dir, img_file), mask_path))
                        mask_found = True
                        break
                
                if mask_found:
                    break
            
            if not mask_found:
                print(f"警告: 未找到图像 {img_file} 对应的mask")
        
        return pairs
    
    def augment_pair(self, image_path, mask_path, augmentations_per_image=4):
        """
        对单对图像和mask进行增强
        
        参数:
            image_path: 原始图像路径
            mask_path: mask图像路径
            augmentations_per_image: 每张图像生成多少增强版本
            
        返回:
            list: 生成的增强文件路径列表
        """
        generated_files = []
        
        try:
            # 读取图像和mask
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # 尝试用PIL读取
                mask_pil = Image.open(mask_path)
                mask = np.array(mask_pil.convert('L'))
            
            if image is None or mask is None:
                print(f"无法读取图像或mask: {image_path}, {mask_path}")
                return generated_files
            
            # 获取基础文件名
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_base_name = os.path.splitext(os.path.basename(mask_path))[0]
            
            # 保存原始文件（可选）
            # self._save_image_mask(image, mask, base_name, mask_base_name, "original")
            
            # 应用不同的增强管道
            for i in range(augmentations_per_image):
                # 随机选择增强管道
                pipeline = random.choice(self.transform_pipelines)
                
                # 应用增强
                augmented = pipeline(image=image, mask=mask)
                aug_image = augmented['image']
                aug_mask = augmented['mask']
                
                # 确保mask仍然是二值图像（0-255）
                if aug_mask.dtype != np.uint8:
                    aug_mask = (aug_mask * 255).astype(np.uint8)
                
                # 生成文件名
                aug_suffix = f"_aug{i+1:03d}"
                aug_img_name = f"{base_name}{aug_suffix}.png"
                aug_mask_name = f"{mask_base_name}{aug_suffix}.png"
                
                # 保存增强后的图像和mask
                aug_img_path = os.path.join(self.img_output_dir, aug_img_name)
                aug_mask_path = os.path.join(self.mask_output_dir, aug_mask_name)
                
                # 保存图像（PNG格式，无损）
                cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(aug_mask_path, aug_mask)
                
                generated_files.append((aug_img_path, aug_mask_path))
            
            return generated_files
            
        except Exception as e:
            print(f"增强失败 {image_path}: {e}")
            return generated_files
    
    def _save_image_mask(self, image, mask, img_base, mask_base, suffix):
        """保存图像和mask"""
        img_path = os.path.join(self.img_output_dir, f"{img_base}_{suffix}.png")
        mask_path = os.path.join(self.mask_output_dir, f"{mask_base}_{suffix}.png")
        
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
        
        return img_path, mask_path
    
    def batch_augment(self, images_dir, masks_dir, augmentations_per_image=4):
        """
        批量增强所有图像
        
        参数:
            images_dir: 原始图像目录
            masks_dir: mask图像目录
            augmentations_per_image: 每张图像生成多少增强版本
            
        返回:
            dict: 增强统计信息
        """
        # 查找图像-mask对
        pairs = self.find_image_mask_pairs(images_dir, masks_dir)
        
        if not pairs:
            print("未找到任何图像-mask对！")
            return {"total_pairs": 0, "augmented": 0}
        
        print(f"找到 {len(pairs)} 对图像-mask")
        print(f"每张图像生成 {augmentations_per_image} 个增强版本")
        print(f"输出目录: {self.output_dir}")
        print("开始增强...")
        
        total_augmented = 0
        
        # 处理每对图像
        for img_path, mask_path in tqdm.tqdm(pairs, desc="增强进度"):
            generated = self.augment_pair(img_path, mask_path, augmentations_per_image)
            total_augmented += len(generated)
            
            # 显示进度
            if len(generated) > 0:
                print(f"  {os.path.basename(img_path)} -> {len(generated)} 个增强版本")
        
        # 复制原始文件到输出目录（可选）
        self._copy_original_files(pairs)
        
        # 统计信息
        original_count = len(pairs)
        augmented_count = total_augmented
        total_count = original_count + augmented_count
        
        print("\n" + "="*50)
        print("增强完成！")
        print(f"原始图像对: {original_count}")
        print(f"增强生成的图像对: {augmented_count}")
        print(f"总计图像对: {total_count}")
        print(f"图像保存位置: {self.img_output_dir}")
        print(f"mask保存位置: {self.mask_output_dir}")
        print("="*50)
        
        return {
            "original_pairs": original_count,
            "augmented_pairs": augmented_count,
            "total_pairs": total_count,
            "images_dir": self.img_output_dir,
            "masks_dir": self.mask_output_dir
        }
    
    def _copy_original_files(self, pairs):
        """复制原始文件到输出目录"""
        for img_path, mask_path in pairs:
            try:
                # 复制原始图像
                img_name = os.path.basename(img_path)
                img_dest = os.path.join(self.img_output_dir, img_name)
                
                # 如果文件不存在才复制
                if not os.path.exists(img_dest):
                    # 读取并保存为PNG（确保格式统一）
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 修改文件扩展名为.png
                        base_name = os.path.splitext(img_name)[0]
                        img_dest_png = os.path.join(self.img_output_dir, f"{base_name}.png")
                        cv2.imwrite(img_dest_png, img)
                
                # 复制原始mask
                mask_name = os.path.basename(mask_path)
                mask_dest = os.path.join(self.mask_output_dir, mask_name)
                
                if not os.path.exists(mask_dest):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        base_name = os.path.splitext(mask_name)[0]
                        mask_dest_png = os.path.join(self.mask_output_dir, f"{base_name}.png")
                        cv2.imwrite(mask_dest_png, mask)
                        
            except Exception as e:
                print(f"复制原始文件失败 {img_path}: {e}")

# 使用示例
if __name__ == "__main__":
    # 安装必要库（如果还没安装）
    # pip install opencv-python numpy pillow albumentations tqdm
    
    # 设置你的数据路径
    REAL_IMAGES_DIR = "E:\\06_Temporary\\data_bridge\\img"  # 原始图像目录
    REAL_MASKS_DIR = "E:\\06_Temporary\\data_bridge\\masks"    # mask目录
    OUTPUT_DIR = "E:\\06_Temporary\\data_bridge_aug"    # 输出目录
    
    # 创建增强器
    augmentor = BridgeDataAugmentor(output_dir=OUTPUT_DIR)
    
    # 执行批量增强
    # 每张图像生成4个增强版本
    stats = augmentor.batch_augment(
        images_dir=REAL_IMAGES_DIR,
        masks_dir=REAL_MASKS_DIR,
        augmentations_per_image=4  # 从100张生成400张
    )
    
    # 如果你想修改增强数量，可以这样调整：
    # augmentations_per_image = 3  # 从100张生成300张
    # augmentations_per_image = 8  # 从100张生成800张
    
    print("\n增强完成！增强后的数据已保存到:")
    print(f"图像: {stats['images_dir']}")
    print(f"Mask: {stats['masks_dir']}")