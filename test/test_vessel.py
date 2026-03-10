from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vesselness2d import *
import time
import os

# --- 1. 配置路径 ---
img_dir = 'VOCdevkit/VOC2007-real/JPEGImages/0001_007.jpg'
mask_dir = 'VOCdevkit/VOC2007-real/SegmentationClass/0001_007.png'

if not os.path.exists(mask_dir):
    print(f"错误：找不到掩码文件 {mask_dir}")
else:
    # --- 2. 读取与预处理 ---
    image = Image.open(img_dir).convert("RGB")
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # ================= 新增：CLAHE 增强 =================
    # clipLimit：对比度限制，值越大对比度越强（建议 2.0-4.0）
    # tileGridSize：网格大小，通常设为 (8,8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image_enhanced = clahe.apply(image_gray)
    # ===================================================

    # 预处理归一化 (使用增强后的图)
    thr = np.percentile(image_enhanced[(image_enhanced > 0)], 1) * 0.9
    image_enhanced_f = image_enhanced.astype(np.float32)
    image_enhanced_f[image_enhanced_f <= thr] = thr
    image_enhanced_f = image_enhanced_f - np.min(image_enhanced_f)
    image_p = image_enhanced_f / (np.max(image_enhanced_f) + 1e-7)

    # --- 3. 读取掩码图 ---
    mask_raw = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask_raw > 0).astype(np.uint8)

    # --- 4. 运行 Vesselness 滤波 ---
    spacing = [1, 1]
    tau = 2
    max_width = 12  
    sigma = np.arange(0.5, (max_width / 2) + 0.5, 0.5).tolist() 

    t1 = time.time()
    vessel_obj = vesselness2d(image_p, sigma, spacing, tau)
    output = vessel_obj.vesselness2d()
    t2 = time.time()

    # --- 5. 形态学后处理：边缘填充 (核心修改) ---
    # 1. 先进行基本的二值化
    output_rescaled = ((output - output.min()) / (output.max() - output.min() + 1e-7) * 255).astype(np.uint8)
    _, binary_output_raw = cv2.threshold(output_rescaled, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 执行闭运算 (Closing) 填充内部空心
    # kernel 大小建议设为 max_width 的一半左右
    fill_kernel_size = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel_size, fill_kernel_size))
    binary_output_filled = cv2.morphologyEx(binary_output_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # --- 6. 计算召回率 (Recall) ---
    # 使用填充后的结果计算
    tp = np.sum((binary_output_filled == 1) & (binary_mask == 1))
    fn = np.sum((binary_output_filled == 0) & (binary_mask == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"消耗时间：{(t2 - t1)*1000:.2f} ms")
    print(f"检测召回率 (Recall): {recall:.4f}")

    # --- 7. 可视化 ---
    plt.figure(figsize=(24, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(image_p, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Vesselness (Raw)")
    plt.imshow(binary_output_raw, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title(f"Filled Result (Recall: {recall:.2f})")
    plt.imshow(binary_output_filled, cmap='jet') 
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Ground Truth (Mask)")
    plt.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.tight_layout()
    plt.show()