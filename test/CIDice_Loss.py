# ============================================
# CLDice 连续 vs 断裂 测试
# ============================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nets.unet_training import CLDice_loss

# ============================================
# 1️⃣  Soft Morphology (用于CLDice)
# ============================================

# def soft_erode(img):
#     p1 = -F.max_pool2d(-img, (3,1), stride=1, padding=(1,0))
#     p2 = -F.max_pool2d(-img, (1,3), stride=1, padding=(0,1))
#     return torch.min(p1, p2)

# def soft_dilate(img):
#     return F.max_pool2d(img, (3,3), stride=1, padding=1)

# def soft_skel(img, iter_):
#     img1 = img
#     skel = torch.zeros_like(img)
#     for _ in range(iter_):
#         img_erode = soft_erode(img1)
#         img_open = soft_dilate(img_erode)
#         delta = F.relu(img1 - img_open)
#         skel = skel + F.relu(delta - skel * delta)
#         img1 = img_erode
#     return skel

# def CLDice_loss(pred, target, iter_=15, smooth=1e-5):
#     skel_pred = soft_skel(pred, iter_)
#     skel_gt = soft_skel(target, iter_)

#     tprec = (skel_pred * target).sum() / (skel_pred.sum() + smooth)
#     trec  = (skel_gt   * pred  ).sum() / (skel_gt.sum() + smooth)

#     cl = (2 * tprec * trec) / (tprec + trec + smooth)
#     return 1 - cl


# ============================================
# 2️⃣  生成测试图像
# ============================================

H, W = 256, 256

# 连续直线 GT
gt = np.zeros((H, W), dtype=np.float32)
gt[120:128, 30:226] = 1.0   # 横向粗线（宽16像素）

# 连续预测（正确情况）
pred_good = gt.copy()

# 断裂预测（人为中间切断）
pred_bad = gt.copy()
pred_bad[120:128, 120:130] = 0

# 严重断裂预测
pref_bad_pro = pred_bad.copy()
pref_bad_pro[120:128, 150:160] = 0

# 转为 tensor
gt_t = torch.tensor(gt).unsqueeze(0).unsqueeze(0)
pred_good_t = torch.tensor(pred_good).unsqueeze(0).unsqueeze(0)
pred_bad_t = torch.tensor(pred_bad).unsqueeze(0).unsqueeze(0)
pred_bad_pro_t = torch.tensor(pref_bad_pro).unsqueeze(0).unsqueeze(0)


# ============================================
# 3️⃣  计算 CLDice Loss
# ============================================

loss_good = CLDice_loss(pred_good_t, gt_t)
loss_bad = CLDice_loss(pred_bad_t, gt_t)
loss_bad_pro = CLDice_loss(pred_bad_pro_t, gt_t)

print("连续预测 CLDice Loss:", float(loss_good))
print("断裂预测 CLDice Loss:", float(loss_bad))
print("严重断裂预测 CLDice Loss:", float(loss_bad_pro))


# ============================================
# 4️⃣  可视化（白色掩码）
# ============================================

# 转为3通道RGB（白色线条）
def to_rgb(img):
    rgb = np.stack([img, img, img], axis=-1)
    return rgb

plt.figure(figsize=(10, 4))

# 左图：连续预测
plt.subplot(1, 3, 1)
plt.title(f"Continuous Prediction\nCLDice Loss: {loss_good:.6f}")
plt.imshow(to_rgb(pred_good))
plt.axis("off")

# 右图：断裂预测
plt.subplot(1, 3, 2)
plt.title(f"Broken Prediction\nCLDice Loss: {loss_bad:.6f}")
plt.imshow(to_rgb(pred_bad))
plt.axis("off")

# 严重断裂预测
plt.subplot(1, 3, 3)
plt.title(f"Pro Broken Prediction\nCLDice Loss: {loss_bad_pro:.6f}")
plt.imshow(to_rgb(pref_bad_pro))
plt.axis("off")

plt.tight_layout()
plt.show()