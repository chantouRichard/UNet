import numpy as np
import cv2

mask = cv2.imread('VOCdevkit/VOC2007/SegmentationClass/1_shot_outer01_inner01.png', cv2.IMREAD_GRAYSCALE)

# 颜色映射
color_map = {
    0: [0, 0, 0],
    1: [0, 0, 255],    # OpenCV是BGR格式，红色是[0,0,255]
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [0, 255, 255],
    5: [255, 0, 255],
}

colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for value, color in color_map.items():
    colored_mask[mask == value] = color

cv2.imshow('Mask', colored_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()