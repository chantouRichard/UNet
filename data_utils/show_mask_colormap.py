import numpy as np
import cv2
from PIL import Image

mask = cv2.imread('temp\\data_aug_2\\masks\\bridge_2_aug004.png', cv2.IMREAD_GRAYSCALE)

# 颜色映射（RGB格式）
color_map = {
    0: (0, 0, 0),       # 黑色
    1: (255, 0, 0),     # 红色
    2: (0, 255, 0),     # 绿色
    3: (0, 0, 255),     # 蓝色
    4: (255, 255, 0),   # 青色
    5: (255, 0, 255),   # 洋红色
}

# 创建彩色mask
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for value, color in color_map.items():
    colored_mask[mask == value] = color

# 转换为PIL图像并显示
colored_pil = Image.fromarray(colored_mask)
colored_pil.show()

# 或者保存查看
# colored_pil.save('colored_mask_pil.png')
# print("彩色mask已保存为 colored_mask_pil.png")