import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr

# ===============================
# 路径
# ===============================
vessel_dir = "Vessel_result"
label_dir = "VOCdevkit/VOC2007/SegmentationClass"

# ===============================
# 统计变量
# ===============================
total_iou = []
total_precision = []
total_recall = []
total_corr = []
total_mi = []

image_list = [f for f in os.listdir(vessel_dir) if f.endswith((".jpg", ".png"))]

print(f"共 {len(image_list)} 张图像")

for name in tqdm(image_list):

    vessel_path = os.path.join(vessel_dir, name)
    label_path = os.path.join(label_dir, name.replace(".jpg", ".png"))

    if not os.path.exists(label_path):
        continue

    # 读取
    vessel = cv2.imread(vessel_path, 0)
    label = cv2.imread(label_path, 0)

    # 二值化
    vessel_bin = (vessel > 128).astype(np.uint8)
    rope_mask = (label == 1).astype(np.uint8)

    # ===============================
    # IoU
    # ===============================
    intersection = np.sum(vessel_bin * rope_mask)
    union = np.sum((vessel_bin + rope_mask) > 0)

    if union == 0:
        continue

    iou = intersection / union
    total_iou.append(iou)

    # ===============================
    # Precision / Recall
    # ===============================
    tp = intersection
    fp = np.sum(vessel_bin) - tp
    fn = np.sum(rope_mask) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    total_precision.append(precision)
    total_recall.append(recall)

    # ===============================
    # 像素级相关
    # ===============================
    vessel_flat = vessel.flatten()
    rope_flat = rope_mask.flatten()

    if np.std(vessel_flat) > 0:
        corr, _ = pearsonr(vessel_flat, rope_flat)
        total_corr.append(corr)

    # ===============================
    # Mutual Information
    # ===============================
    mi = mutual_info_score(rope_flat, vessel_flat > 128)
    total_mi.append(mi)

# ===============================
# 输出结果
# ===============================
print("\n========== 统计结果 ==========")
print(f"平均 IoU            : {np.mean(total_iou):.4f}")
print(f"平均 Precision      : {np.mean(total_precision):.4f}")
print(f"平均 Recall         : {np.mean(total_recall):.4f}")
print(f"平均 Pearson Corr   : {np.mean(total_corr):.4f}")
print(f"平均 Mutual Info    : {np.mean(total_mi):.4f}")
print("================================")