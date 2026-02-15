import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nets.unet_training import CLDice_loss
from utils.cldice import soft_cldice, soft_dice, soft_dice_cldice
# ==============================
# 1. 读取图片
# ==============================

# pred_path = r"miou_out/miou_2026_02_11_20_52_28-CBAM+50/detection-results/0002_007.png"
pred_path = r"miou_out/miou_2026_02_15_16_08_53/detection-results/0001_004.png"
gt_path   = r"VOCdevkit/VOC2007-real/SegmentationClass/0001_004.png"

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

cldice_loss_fn = soft_cldice(iter_=3)
dice_cldice_fn = soft_dice_cldice(iter_=3, alpha=0.5)

loss = soft_dice(gt_tensor, pred_tensor)
loss_value = loss.item()

cl_loss = cldice_loss_fn(gt_tensor, pred_tensor)
dice_loss = soft_dice(gt_tensor, pred_tensor)
dice_cl_loss = dice_cldice_fn(gt_tensor, pred_tensor)

print("Soft Dice Loss:", dice_loss.item())
print("Soft CLDice Loss:", cl_loss.item())
print("Soft Dice + CLDice Loss:", dice_cl_loss.item())
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

# import torch
# import numpy as np
# import cv2
# import os
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from utils.cldice import soft_cldice, soft_dice, soft_dice_cldice

# # 预测文件夹
# pred_dir = r"miou_out/miou_2026_02_14_22_59_34/detection-results"
# gt_dir   = r"VOCdevkit/VOC2007-real/SegmentationClass"

# for filename in os.listdir(pred_dir):
#     if not filename.endswith(".png"):
#         continue

#     pred_path = os.path.join(pred_dir, filename)
#     gt_path   = os.path.join(gt_dir, filename)

#     if not os.path.exists(gt_path):
#         continue

#     # 读取
#     pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
#     gt_img   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

#     # 二值化
#     pred_bin = (pred_img > 0).astype(np.float32)
#     gt_bin   = (gt_img > 0).astype(np.float32)

#     # tensor
#     pred_tensor = torch.from_numpy(pred_bin).unsqueeze(0).unsqueeze(0)
#     gt_tensor   = torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0)

#     print(f"处理{filename}...")
#     # 计算 CLDice
#     loss = soft_dice(pred_tensor, gt_tensor)

#     if loss < 0:
#         print(f"负数CLDice: {filename}  ->  {loss:.6f}")