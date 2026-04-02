import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm

def visualize_segmentation(image_path, mask_path, save_dir=None, show=True, alpha=0.5):
    """
    可视化单张图像和掩码的叠加效果
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告: 无法读取图像 {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"警告: 无法读取掩码 {mask_path}")
        return

    # 检查尺寸是否一致，不一致则缩放掩码（防止UNet输出尺寸偏移）
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建彩色掩码（红色）
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask > 0] = [255, 255, 255]  # 这里可改为 [0, 255, 0] 绿色

    # 混合显示
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, colored_mask, alpha, 0)

    # 处理保存逻辑
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, Path(image_path).name)
        # cv2保存需要转回BGR
        save_img = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)

    # 处理显示逻辑
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(overlay)
        plt.title(f"Visualizing: {Path(image_path).name}")
        plt.axis('off')
        plt.show()

def batch_visualize(img_folder, mask_folder, save_dir=None, show=False):
    """
    批量处理文件夹
    """
    img_list = sorted([f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"开始处理文件夹，共找到 {len(img_list)} 张图片...")
    for img_name in tqdm(img_list, disable=True):
        # 假设掩码和原图同名，但可能后缀不同（如 .png）
        pure_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_folder, img_name)
        
        # 匹配掩码路径（尝试 .png 或 .jpg）
        mask_path = os.path.join(mask_folder, pure_name + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_folder, pure_name + ".jpg")

        visualize_segmentation(img_path, mask_path, save_dir=save_dir, show=show)

# --- 使用示例 ---

# 1. 单张图片可视化 (只显示不保存)
visualize_segmentation(
    image_path=r"VOCdevkit\\VOC2007\\JPEGImages\\0001_000.jpg",
    mask_path=r"VOCdevkit\\VOC2007\\SegmentationClass\\0001_000.png",
    save_dir=None, 
    show=True,
    alpha=1.0
)

# 2. 批量处理并保存 (不显示，直接存入目标文件夹)
# IMG_DIR = r"E:\06_Temporary\data_bridge_3\img"
# MASK_DIR = r"E:\06_Temporary\data_bridge_3\masks"
# OUT_DIR = r"E:\06_Temporary\data_bridge_3\previews"

# batch_visualize(IMG_DIR, MASK_DIR, save_dir=OUT_DIR, show=False)