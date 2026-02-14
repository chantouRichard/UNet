import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nets.unet_training import CLDice_loss
# ==============================
# 1. 读取图片
# ==============================

# pred_path = r"miou_out/miou_2026_02_11_20_52_28-CBAM+50/detection-results/0002_007.png"
pred_path = r"miou_out/miou_2026_02_14_22_59_34/detection-results/0002_007.png"
gt_path   = r"VOCdevkit/VOC2007-real/SegmentationClass/0002_007.png"

pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
gt_img   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

# ==============================
# 2. 归一化到0/1
# ==============================

pred_bin = (pred_img > 0).astype(np.float32)
gt_bin   = (gt_img > 0).astype(np.float32)

# 转成 tensor
pred_tensor = torch.from_numpy(pred_bin).unsqueeze(0).unsqueeze(0)
gt_tensor   = torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0)

# ==============================
# 3. 计算 CLDice
# ==============================

loss = CLDice_loss(pred_tensor, gt_tensor)
loss_value = loss.item()

print("CLDice Loss:", loss_value)

# ==============================
# 4. 1 → 255 变白色
# ==============================

pred_show = pred_bin.copy()
gt_show   = gt_bin.copy()

pred_show[pred_show == 1] = 255
gt_show[gt_show == 1] = 255

# ==============================
# 5. 可视化
# ==============================

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Prediction")
plt.imshow(pred_show, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title(f"Ground Truth\nCLDice Loss: {loss_value:.4f}")
plt.imshow(gt_show, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()