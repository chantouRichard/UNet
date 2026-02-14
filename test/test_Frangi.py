from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Frangi import frangi_filter
# ==============================
# 读取图像
# ==============================
img = Image.open("img/AI-3.png").convert("L")
img = np.array(img) / 255.0

# 可选轻微平滑
img = cv2.GaussianBlur(img, (3,3), 0.5)

# ==============================
# 运行 Frangi
# ==============================
vessel = frangi_filter(
    img,
    scale_range=(1, 10),
    scale_step=1,
    beta1=0.5,
    beta2=15,
    black_ridges=False
)

# 二值化
_, vessel_bin = cv2.threshold(
    (vessel * 255).astype(np.uint8),
    30, 255, cv2.THRESH_BINARY
)

# ==============================
# 显示结果
# ==============================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Frangi Response")
plt.imshow(vessel, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Binary Result")
plt.imshow(vessel_bin, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()