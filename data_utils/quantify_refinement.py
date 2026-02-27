import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. 路径配置 ---
pred_dir = r'miou_out\miou_2026_02_11_20_52_28-CBAM+50\detection-results'
gt_dir = r'VOCdevkit\VOC2007\SegmentationClass'
vessel_dir = r'Vessel_result'

# --- 2. 后处理超参数 (你可以微调这两个参数) ---
HESSIAN_THRESH = 80  # Hessian响应的阈值，建议设在40-60之间
MIN_OVERLAP = 10      # 至少有多少个像素重叠才“救回”该区域

def calculate_iou(pd, gt):
    intersection = np.logical_and(pd, gt).sum()
    union = np.logical_or(pd, gt).sum()
    if union == 0: return 1.0
    return intersection / union

def run_quantify_demo():
    file_list = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    old_ious = []
    new_ious = []

    print(f"开始量化实验，共 {len(file_list)} 张图片...")

    for file_name in tqdm(file_list):
        base_name = os.path.splitext(file_name)[0]
        
        # 路径拼接
        pred_path = os.path.join(pred_dir, file_name)
        gt_path = os.path.join(gt_dir, base_name + ".png")
        vessel_path = os.path.join(vessel_dir, base_name + ".jpg")

        if not (os.path.exists(gt_path) and os.path.exists(vessel_path)):
            continue

        # 3. 读取图像并二值化
        pred_img = cv2.imread(pred_path, 0)
        gt_img = cv2.imread(gt_path, 0)
        vessel_img = cv2.imread(vessel_path, 0)

        # 统一到 0/1 尺度计算 IoU
        gt_mask = (gt_img == 1).astype(np.uint8)
        if pred_img.max() > 1:
            pred_mask = (pred_img > 127).astype(np.uint8)
        else:
            pred_mask = pred_img.astype(np.uint8)

        # 4. 后处理核心算法：基于连通域的修复
        # a. 提取Hessian候选区域
        _, hessian_bin = cv2.threshold(vessel_img, HESSIAN_THRESH, 255, cv2.THRESH_BINARY)
        
        # b. 对Hessian二值图进行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hessian_bin)
        
        refined_mask = pred_mask.copy()
        
        for i in range(1, num_labels):
            # 提取第i个连通块
            component = (labels == i).astype(np.uint8)
            
            # 计算该连通块与原预测图的交集像素点数
            overlap = np.logical_and(component, pred_mask).sum()
            
            # 核心判断：如果Hessian连通块与预测图有重叠，说明这是绳索的延伸，救回它
            if overlap >= MIN_OVERLAP:
                refined_mask[labels == i] = 1

        # 5. 计算 IoU
        old_iou = calculate_iou(pred_mask, gt_mask)
        new_iou = calculate_iou(refined_mask, gt_mask)
        
        old_ious.append(old_iou)
        new_ious.append(new_iou)

    # 6. 输出结果
    print("\n" + "="*40)
    print(f"实验参数: Hessian阈值={HESSIAN_THRESH}, 重叠像素={MIN_OVERLAP}")
    print(f"原始平均 IoU: {np.mean(old_ious):.4f}")
    print(f"后处理后平均 IoU: {np.mean(new_ious):.4f}")
    print(f"IoU 绝对提升: {(np.mean(new_ious) - np.mean(old_ious))*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_quantify_demo()