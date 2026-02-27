import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 路径配置 ---
pred_dir = r'test_JPEG'
gt_dir = r'VOCdevkit\VOC2007\SegmentationClass'
vessel_dir = r'Vessel_result'

def calculate_iou(pd, gt):
    intersection = np.logical_and(pd, gt).sum()
    union = np.logical_or(pd, gt).sum()
    if union == 0: return 1.0
    return intersection / union

def run_oracle_experiment():
    file_list = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    old_ious = []
    oracle_ious = [] # 理想实验IoU

    print("开始理想实验（验证Hessian特征上限）...")

    for file_name in tqdm(file_list):
        base_name = os.path.splitext(file_name)[0]
        
        # 路径
        pred_path = os.path.join(pred_dir, base_name + ".jpg")
        gt_path = os.path.join(gt_dir, base_name + ".png")
        vessel_path = os.path.join(vessel_dir, base_name + ".jpg")

        if not (os.path.exists(gt_path) and os.path.exists(vessel_path)):
            continue

        # 读取
        img_pred = cv2.imread(pred_path, 0)
        img_gt = cv2.imread(gt_path, 0)
        img_vessel = cv2.imread(vessel_path, 0)

        # 统一尺度
        gt_mask = (img_gt == 1).astype(np.uint8)
        if img_pred.max() > 1:
            pred_mask = (img_pred > 127).astype(np.uint8)
        else:
            pred_mask = img_pred.astype(np.uint8)

        # --- 理想实验核心逻辑 ---
        # 1. 找到网络漏掉的区域 (FN)
        fn_mask = cv2.subtract(gt_mask, pred_mask)
        fn_mask[fn_mask < 0] = 0
        
        # 2. 在这些漏掉的区域里，看看Hessian有没有响应
        # 我们设定一个合理的Hessian阈值，比如50
        hessian_candidates = (img_vessel > 50).astype(np.uint8)
        
        # 3. 理想修补：只把属于GT范围内的Hessian响应补回去
        # 这样就完全排除了背景噪声
        useful_hessian = cv2.bitwise_and(hessian_candidates, gt_mask)
        
        # 4. 生成理想修复图
        oracle_refined = cv2.bitwise_or(pred_mask, useful_hessian)

        # 计算 IoU
        old_ious.append(calculate_iou(pred_mask, gt_mask))
        oracle_ious.append(calculate_iou(oracle_refined, gt_mask))

    print("\n" + "="*40)
    print(f"原始平均 IoU: {np.mean(old_ious):.4f}")
    print(f"理想实验(无噪声补全) IoU: {np.mean(oracle_ious):.4f}")
    print(f"理论最高提升潜力: {(np.mean(oracle_ious) - np.mean(old_ious))*100:.2f}%")
    print("="*40)

    if np.mean(oracle_ious) > np.mean(old_ious):
        print("结论：Hessian确实包含有效信息！只要解决噪声过滤问题，就能提分。")
    else:
        print("结论：即便排除噪声，Hessian也没能提供更多GT信息，可能需要调整Hessian算法参数。")

if __name__ == "__main__":
    run_oracle_experiment()