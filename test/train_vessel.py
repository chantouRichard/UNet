import os
import cv2
import numpy as np
import time
import optuna
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from vesselness2d import *

# --- 1. 数据加载函数 ---
def load_data(img_root, mask_root, num_samples=20):
    """从VOC目录加载一定数量的样本进行优化"""
    img_files = sorted(os.listdir(img_root))[:num_samples]
    data_pairs = []
    
    for f in img_files:
        img_path = os.path.join(img_root, f)
        mask_path = os.path.join(mask_root, f.replace('.jpg', '.png'))
        
        if os.path.exists(mask_path):
            # 预处理原图
            img = np.array(Image.open(img_path).convert("L"))
            img_p = (img - img.min()) / (img.max() - img.min() + 1e-7)
            
            # 预处理掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            binary_mask = (mask > 0).astype(np.uint8)
            
            data_pairs.append((img_p, binary_mask))
    return data_pairs

# --- 2. 评价指标计算 ---
def calculate_metrics(pred, gt):
    # 计算 IoU
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-7)
    
    # 计算 Recall
    tp = intersection
    fn = np.logical_and(pred == 0, gt == 1).sum()
    recall = tp / (tp + fn + 1e-7)
    
    return iou, recall

# --- 3. 优化目标函数 ---
def objective(trial, data_pairs):
    # 自动定义搜索空间 (无需预设数组)
    max_width = trial.suggest_float("max_width", 1.0, 30.0)
    fill_kernel_size = trial.suggest_int("fill_kernel_size", 3, 21, step=2) # 必须为奇数
    
    total_miou = 0
    total_recall = 0
    
    # 模拟“训练”过程：对数据样本进行处理
    for img_p, gt in data_pairs:
        # 1. 动态生成 Sigma
        sigma = np.arange(0.5, (max_width / 2) + 0.5, 0.5).tolist()
        
        # 2. Vesselness 滤波
        vessel_obj = vesselness2d(img_p, sigma, [1, 1], tau=2)
        output = vessel_obj.vesselness2d()
        
        # 3. 后处理：填充
        output_rescaled = ((output - output.min()) / (output.max() - output.min() + 1e-7) * 255).astype(np.uint8)
        _, binary_raw = cv2.threshold(output_rescaled, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel_size, fill_kernel_size))
        binary_filled = cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, kernel)
        
        # 4. 指标累计
        iou, recall = calculate_metrics(binary_filled, gt)
        total_miou += iou
        total_recall += recall
        
    avg_miou = total_miou / len(data_pairs)
    avg_recall = total_recall / len(data_pairs)
    
    # 目标：我们希望最大化 mIoU 和 Recall 的加权得分
    return 0.2 * avg_miou + 0.8 * avg_recall

# --- 4. 主程序 ---
if __name__ == "__main__":
    img_dir = 'VOCdevkit/VOC2007-real/JPEGImages'
    mask_dir = 'VOCdevkit/VOC2007-real/SegmentationClass'
    
    print("正在加载数据样本...")
    samples = load_data(img_dir, mask_dir, num_samples=15) # 取15张代表性图片进行寻优
    
    # 创建 Optuna 研究对象
    # direction="maximize" 表示我们要找得分最大的参数
    study = optuna.create_study(direction="maximize")
    
    print("开始自动寻优参数 (Bayesian Optimization)...")
    pbar = tqdm(total=50) # 进行50次迭代尝试
    
    def callback(study, trial):
        pbar.update(1)
        pbar.set_description(f"Best Score: {study.best_value:.4f}")

    study.optimize(lambda t: objective(t, samples), n_trials=50, callbacks=[callback])
    pbar.close()

    # 输出结果
    print("\n" + "="*30)
    print("寻优结束！")
    print(f"最佳参数: {study.best_params}")
    print(f"最佳综合得分: {study.best_value:.4f}")
    print("="*30)

    # 提取结果以便后续使用
    best_w = study.best_params["max_width"]
    best_k = study.best_params["fill_kernel_size"]