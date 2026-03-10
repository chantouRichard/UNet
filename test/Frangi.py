import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

# =========================
# Fast Vesselness (GPU版)
# =========================
def fast_vesselness(x, dilation=1):
    # x: (B,1,H,W)

    sobel_xx = torch.tensor([[1,-2,1],
                             [2,-4,2],
                             [1,-2,1]], dtype=torch.float32)

    sobel_yy = sobel_xx.t()

    sobel_xy = torch.tensor([[1,0,-1],
                             [0,0,0],
                             [-1,0,1]], dtype=torch.float32)

    sobel_xx = sobel_xx.view(1,1,3,3).to(x.device)
    sobel_yy = sobel_yy.view(1,1,3,3).to(x.device)
    sobel_xy = sobel_xy.view(1,1,3,3).to(x.device)

    Ixx = F.conv2d(x, sobel_xx, padding=dilation, dilation=dilation)
    Iyy = F.conv2d(x, sobel_yy, padding=dilation, dilation=dilation)
    Ixy = F.conv2d(x, sobel_xy, padding=dilation, dilation=dilation)

    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy**2

    response = torch.relu(det - 0.25 * trace**2)

    return response


# =========================
# 读取图像
# =========================
img_real = Image.open("img/AI-3.png").convert("L")
img_real_origin = img_real
img_real = np.array(img_real) / 255.0
img_real = cv2.GaussianBlur(img_real, (3,3), 0.5)

# 转 torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_tensor = torch.from_numpy(img_real).float().unsqueeze(0).unsqueeze(0).to(device)

# =========================
# 多尺度（用 dilation 模拟）
# =========================
v1 = fast_vesselness(img_tensor, dilation=1)
v2 = fast_vesselness(img_tensor, dilation=2)
v3 = fast_vesselness(img_tensor, dilation=3)
v4 = fast_vesselness(img_tensor, dilation=4)

vessel = torch.max(torch.max(v1, v2), torch.max(v3, v4))

# 转回 numpy
vessel = vessel.squeeze().detach().cpu().numpy()

# 二值化
_, vessel_bin = cv2.threshold(
    (vessel * 255).astype(np.uint8),
    30, 255, cv2.THRESH_BINARY
)

# =========================
# 显示结果
# =========================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_real_origin, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Fast Vesselness")
plt.imshow(vessel, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Thresholded")
plt.imshow(vessel_bin, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()