import cv2
import numpy as np
import os

# ------------------- 配置部分 -------------------
# 1. 填入你想要测试的单张掩码图路径 (可以是虚拟GT，也可以是预测图)
# 示例：r"VOCdevkit\VOC2007-virtual\SegmentationClass\000001.png"
TEST_MASK_PATH = r"VOCdevkit\VOC2007-virtual\SegmentationClass\0001_009.png" 

# 2. 设置想要提取的目标像素值 (学长说绳索是1)
TARGET_ID = 1
# ------------------------------------------------

def test_single_mask(image_path, target_id):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到文件，请检查路径是否正确：\n{image_path}")
        return

    # 1. 以灰度模式读取图片
    # 使用 cv2.IMREAD_GRAYSCALE 确保读取的是单通道灰度图
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"❌ 错误：无法读取图片，文件可能损坏。")
        return

    print(f"成功读取图片，原始尺寸: {mask.shape}")
    print(f"原始图片像素最大值: {mask.max()}, 最小值: {mask.min()}")

    # 2. 核心可视化逻辑：创建二值化掩码
    # np.where(condition, x, y): 
    # 如果像素值等于 target_id，则设为 255 (白色)
    # 否则设为 0 (黑色)
    # 这样就只把像素为 1 的绳索“亮”出来了
    visualization = np.where(mask == target_id, 255, 0).astype(np.uint8)

    # 3. 显示结果
    # 创建窗口并调整大小以便查看
    cv2.namedWindow('Original Mask (Raw)', cv2.WINDOW_NORMAL)
    cv2.namedWindow(f'Extracted Target (ID={target_id})', cv2.WINDOW_NORMAL)
    
    # 显示原始图（通常看起来是全黑的，因为像素值太小）
    cv2.imshow('Original Mask (Raw)', mask)
    # 显示提取后的可视化图
    cv2.imshow(f'Extracted Target (ID={target_id})', visualization)

    print("\n✅ 已生成可视化窗口。")
    print(" - 原始图通常看起来是黑色的，因为像素值 1 在 0-255 范围内非常暗。")
    print(" - 可视化图中纯白色的部分就是成功提取出的绳索。")
    print("\n按任意键关闭窗口并退出程序...")
    
    cv2.waitKey(0) # 等待按键
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 你可以直接修改上面的 TEST_MASK_PATH 变量，然后运行此脚本
    test_single_mask(TEST_MASK_PATH, TARGET_ID)