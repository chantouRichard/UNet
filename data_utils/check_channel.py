import cv2
import numpy as np


def check_image_channels(image_path):
    """
    快速检查图片通道信息
    """
    # 读取图片（保持原样，不转换）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    print(f"图片: {image_path}")
    print(f"Shape: {img.shape}")
    print(f"数据类型: {img.dtype}")
    print(f"最小像素值: {img.min()}")
    print(f"最大像素值: {img.max()}")
    print(f"唯一像素值数量: {len(np.unique(img))}")

    # 判断通道数
    if len(img.shape) == 2:
        print("✅ 这是灰度图 (单通道)")
    elif len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 3:
            print("🎨 这是三通道彩色图 (BGR)")
            # 检查是否实际上是灰度图存成了三通道
            if np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 0], img[:, :, 2]):
                print("  注意：虽然是三通道，但三个通道值相同，实际上是灰度图")
            else:
                # 显示一些示例像素值
                print(f"  示例像素值(前3个): {img[0, 0, :]}")
        elif channels == 4:
            print("🎨 这是四通道图 (RGBA/BGRA，包含透明度)")
        elif channels == 1:
            print("✅ 这是单通道图 (shape中有冗余的1)")
    else:
        print("❓ 未知格式")

    return img


# 使用示例
image_path = "VOCdevkit/VOC2007/SegmentationClass/3_shot_outer01_inner01.png"
img = check_image_channels(image_path)