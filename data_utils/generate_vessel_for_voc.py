import os
import time
import numpy as np
import cv2
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.vesselness2d import vesselness2d

# ===============================
# 路径设置
# ===============================
input_dir = "VOCdevkit/VOC2007/JPEGImages"
output_dir = "Vessel_result"

os.makedirs(output_dir, exist_ok=True)

# ===============================
# Vesselness 参数
# ===============================
sigma = [0.5, 1, 2, 3, 4, 5]
spacing = [1, 1]
tau = 2

# ===============================
# 获取所有图片
# ===============================
image_list = [f for f in os.listdir(input_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"共发现 {len(image_list)} 张图片")

total_start = time.time()

# ===============================
# 批量处理
# ===============================
from tqdm import tqdm
for idx, img_name in enumerate(tqdm(image_list, total=len(image_list), desc="Processing")):
    img_path = os.path.join(input_dir, img_name)

    # 读取图片
    image = Image.open(img_path).convert("RGB")
    image = np.array(image)

    # 转灰度（必须先灰度）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 确保是 uint8
    image = image.astype(np.uint8)

    # ===============================
    # ① CLAHE 增强
    # ===============================
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # image = clahe.apply(image)

    # ===============================
    # ② 高通增强（强化细结构）
    # ===============================
    # blur = cv2.GaussianBlur(image, (0,0), 3)
    # image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

    # ===============================
    # ③ 转 float 归一化
    # ===============================
    image = image.astype(np.float32)
    image = image - np.min(image)
    image = image / (np.max(image) + 1e-8)

    # 计算 vesselness
    start = time.time()
    vessel = vesselness2d(image, sigma, spacing, tau)
    output = vessel.vesselness2d()
    end = time.time()

    # 归一化到 0-255 保存
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)
    output = (output * 255).astype(np.uint8)

    # 保存
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, output)

    # print(f"[{idx+1}/{len(image_list)}] {img_name} 处理完成 "
    #       f"({(end-start)*1000:.2f} ms)")

total_end = time.time()

print("\n全部处理完成")
print(f"总耗时: {(total_end - total_start):.2f} 秒")