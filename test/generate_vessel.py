import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from vesselness2d import *

# --- 1. 配置路径 ---
# 请确保路径指向你的 VOC 文件夹
IMG_DIR = r'VOCdevkit\VOC2007-real\JPEGImages'
SAVE_DIR = r'VOCdevkit\VOC2007-real\VesselMasks'

# --- 2. 算法核心函数 ---
def process_single_image(img_name):
    """
    处理单张图像的逻辑，封装了增强、滤波和形态学填充
    """
    try:
        img_path = os.path.join(IMG_DIR, img_name)
        save_path = os.path.join(SAVE_DIR, img_name.replace('.jpg', '.png'))
        
        # 如果已经存在则跳过（方便断点续传）
        if os.path.exists(save_path):
            return True

        # 读取灰度图
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: return False

        # A. CLAHE 增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_gray)
        
        # B. 归一化
        img_f = img_enhanced.astype(np.float32)
        img_p = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-7)

        # C. Vesselness 滤波 (使用你测试好的参数)
        spacing = [1, 1]
        tau = 2
        max_width = 12
        sigma = np.arange(0.5, (max_width / 2) + 0.5, 0.5).tolist()
        
        vessel_obj = vesselness2d(img_p, sigma, spacing, tau)
        output = vessel_obj.vesselness2d()

        # D. 后处理：二值化与闭运算填充
        # 归一化到 0-255
        output_res = ((output - output.min()) / (output.max() - output.min() + 1e-7) * 255).astype(np.uint8)
        # Otsu 二值化
        _, bin_raw = cv2.threshold(output_res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 闭运算填充 (使用你测试好的 25 大小核)
        fill_kernel_size = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel_size, fill_kernel_size))
        bin_filled = cv2.morphologyEx(bin_raw, cv2.MORPH_CLOSE, kernel)

        # E. 保存结果
        cv2.imwrite(save_path, bin_filled)
        return True
    except Exception as e:
        print(f"处理 {img_name} 出错: {e}")
        return False

# --- 3. 主程序 ---
if __name__ == "__main__":
    # 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建目录: {SAVE_DIR}")

    # 获取所有 jpg 文件名
    all_imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]
    print(f"找到 {len(all_imgs)} 张图像，准备开始生成...")

    # 使用多进程池加速 (根据 CPU 核心数自动分配，建议留 1-2 个核心)
    # 如果内存不足，可以手动指定 processes=4
    num_processors = max(1, os.cpu_count() - 2) 
    
    with Pool(processes=num_processors) as pool:
        # 使用 tqdm 显示进度
        list(tqdm(pool.imap(process_single_image, all_imgs), total=len(all_imgs), desc="生成 VesselMasks"))

    print("\n所有先验掩码生成完毕！")