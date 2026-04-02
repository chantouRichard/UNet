import os
import cv2
import numpy as np
from PIL import Image

# ------------------- 全局变量配置 -------------------
# 1. 想要挑选展示的图片名（不要后缀）
TARGET_NAMES = ['0122_001', '0037_011', '0008_004', '167_001']

# 2. 文件夹路径配置
# 输入：原图与GT
IMG_ROOT = r'VOCdevkit\VOC2007\JPEGImages'
GT_ROOT = r'VOCdevkit\VOC2007\SegmentationClass'

# 输入：模型预测图所在的根目录
MIOU_BASE = 'miou_out'
MODELS = {
    'BaseLine': 'miou_unet_100epoch',
    'AttUnet': 'miou_attunet',
    'DeepLabv3Plus': 'miou_deeplabv3plus',
    'UnetPlusPlus': 'miou_unetplusplus',
    'Ours': 'miou_unet_dir_cbam'
}

# 3. 输出目标文件夹
SAVE_ROOT = 'comparison_results'

# ---------------------------------------------------

def process_and_save():
    # 创建保存的总目录
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    for img_name in TARGET_NAMES:
        print(f"正在处理图片: {img_name}")
        
        # --- A. 处理原图 ---
        src_img_path = os.path.join(IMG_ROOT, img_name + '.jpg')
        if os.path.exists(src_img_path):
            img = cv2.imread(src_img_path)
            save_path = os.path.join(SAVE_ROOT, f"{img_name}_Image.png")
            cv2.imwrite(save_path, img)

        # --- B. 处理GT (通常需要转换) ---
        gt_path = os.path.join(GT_ROOT, img_name + '.png')
        if os.path.exists(gt_path):
            # 使用PIL读取是因为VOC的GT通常是P模式（调色板）
            # gt = Image.open(gt_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)  # 直接读取原始数据
            if gt.max() < 10:  # 如果GT图像是类别ID形式，扩展到255
                gt = (gt * (255 // (gt.max() if gt.max() > 0 else 1))).astype(np.uint8)
            
            cv2.imwrite(os.path.join(SAVE_ROOT, f"{img_name}_GT.png"), gt)  # 直接保存为PNG格式
            # 转换为RGB便于直接观察对比

        # --- C. 处理各个模型的预测图 ---
        for model_nick, folder_name in MODELS.items():
            pred_path = os.path.join(MIOU_BASE, folder_name, "detection-results", img_name + '.png')
            
            if os.path.exists(pred_path):
                # 读取预测图（通常是单通道灰度图）
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                
                # 转换逻辑：如果你的预测图里只有0, 1, 2等类ID，需要扩充到0-255
                # 这里假设你需要将非背景区域变亮，如果是多类别建议用调色板映射
                # 简单粗暴法：如果最大值很小（如1, 2），则映射到255；如果是已经算好的则保持
                if pred.max() < 10: 
                    # 这里假设是简单的二分类映射，如果是多分类，建议查找调色板
                    pred = (pred * (255 // (pred.max() if pred.max() > 0 else 1))).astype(np.uint8)
                
                # 保存
                save_name = f"{img_name}_{model_nick}.png"
                cv2.imwrite(os.path.join(SAVE_ROOT, save_name), pred)
            else:
                print(f"  [Warning] 找不到路径: {pred_path}")

    print("-" * 30)
    print(f"处理完成！所有对比图已保存在: {SAVE_ROOT}")

if __name__ == '__main__':
    process_and_save()