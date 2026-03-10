import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from vesselness2d import *

# --- 1. 配置路径 ---
img_root = 'VOCdevkit/VOC2007-real/JPEGImages'
mask_root = 'VOCdevkit/VOC2007-real/SegmentationClass'

# --- 2. 参数配置 (沿用你之前的最佳配置) ---
max_width = 12
fill_kernel_size = 25
spacing = [1, 1]
tau = 2
sigma = np.arange(0.5, (max_width / 2) + 0.5, 0.5).tolist()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel_size, fill_kernel_size))

# 获取文件列表
img_names = [f for f in os.listdir(img_root) if f.endswith('.jpg')]
recall_list = []

print(f"开始批量处理 {len(img_names)} 张图片...")
pbar = tqdm(img_names)

for name in pbar:
    img_path = os.path.join(img_root, name)
    mask_path = os.path.join(mask_root, name.replace('.jpg', '.png'))
    
    if not os.path.exists(mask_path):
        continue

    # --- 图像增强与预处理 ---
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_enhanced = clahe.apply(img_gray)
    
    # 归一化
    thr = np.percentile(img_enhanced[img_enhanced > 0], 1) * 0.9
    img_f = img_enhanced.astype(np.float32)
    img_f[img_f <= thr] = thr
    img_f = img_f - np.min(img_f)
    img_p = img_f / (np.max(img_f) + 1e-7)

    # --- Vesselness 滤波 ---
    vessel_obj = vesselness2d(img_p, sigma, spacing, tau)
    output = vessel_obj.vesselness2d()

    # --- 形态学填充 ---
    output_res = ((output - output.min()) / (output.max() - output.min() + 1e-7) * 255).astype(np.uint8)
    _, bin_raw = cv2.threshold(output_res, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_filled = cv2.morphologyEx(bin_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # --- 计算 Recall ---
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask_raw > 0).astype(np.uint8)
    
    tp = np.sum((bin_filled == 1) & (binary_mask == 1))
    fn = np.sum((bin_filled == 0) & (binary_mask == 1))
    
    current_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_list.append(current_recall)
    
    # 更新进度条显示的平均 Recall
    pbar.set_description(f"Avg Recall: {np.mean(recall_list):.4f}")

print(f"\n最终平均召回率 (Mean Recall): {np.mean(recall_list):.4f}")