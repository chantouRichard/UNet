from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vesselness2d import *

img_dir = 'img/0004_001.jpg' #路径写自己的

#reading image
image = Image.open(img_dir).convert("RGB")
image = np.array(image)
# plt.figure(figsize=(10,10))
# plt.imshow(image, cmap='gray')

#convert forgeground to background and vice-versa
# image = 255-image

print(f"图片尺寸：")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thr = np.percentile(image[(image > 0)], 1)*0.9
image[(image <= thr)] = thr
image = image - np.min(image)
image = image / np.max(image)

sigma=[0.5,1, 1.5, 2, 2.5]
spacing = [1, 1]
tau = 2

import time

t1 = time.time()

output = vesselness2d(image, sigma, spacing, tau)
output = output.vesselness2d()

t2 = time.time()

print(f"消耗时间：{(t2 - t1)*1000:.2f} ms")

print("输出结果")
plt.figure(figsize=(16,8))

# 原图
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Vesselness结果
plt.subplot(1,2,2)
plt.title("Vesselness")
plt.imshow(output, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()