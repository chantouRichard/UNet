import os
import shutil
from PIL import Image
import numpy as np
import cv2


def process_images():
    # 源目录
    source_dir = r"D:\04_Media\Downloads\data"
    # 目标目录
    target_jpeg_dir = r"../VOCdevkit/VOC2007/JPEGImages"
    target_seg_dir = r"../VOCdevkit/VOC2007/SegmentationClass"

    # 创建目标目录
    os.makedirs(target_jpeg_dir, exist_ok=True)
    os.makedirs(target_seg_dir, exist_ok=True)

    # 目标尺寸：宽1920，高1920
    target_width = 1920
    target_height = 1920
    original_height = 1024

    # 需要填充的像素数
    padding_height = target_height - original_height
    top_padding = padding_height // 2
    bottom_padding = padding_height - top_padding

    # 处理3~8文件夹
    for folder_num in range(3, 9):
        folder_name = str(folder_num)
        folder_path = os.path.join(source_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"警告：目录 {folder_path} 不存在，跳过")
            continue

        print(f"正在处理文件夹: {folder_name}")

        # 处理原图
        img_source_dir = os.path.join(folder_path, "img")
        if os.path.exists(img_source_dir):
            img_files = [f for f in os.listdir(img_source_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

            for img_file in img_files:
                img_path = os.path.join(img_source_dir, img_file)

                try:
                    # 打开图像
                    img = Image.open(img_path)

                    # 转换为RGB（确保3通道）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # 创建新的1920x1920黑色背景图像
                    new_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))

                    # 将原始图像粘贴到中间
                    new_img.paste(img, (0, top_padding))

                    # 保存图像
                    base_name = os.path.splitext(img_file)[0]
                    target_path = os.path.join(target_jpeg_dir, f"{folder_name}_{base_name}.jpg")
                    new_img.save(target_path, 'JPEG', quality=95)

                    print(f"  已处理原图: {img_file} -> {folder_name}_{base_name}.jpg")

                except Exception as e:
                    print(f"  处理原图 {img_file} 时出错: {e}")

        # 处理掩码图
        mask_source_dir = os.path.join(folder_path, "masks")
        if os.path.exists(mask_source_dir):
            mask_files = [f for f in os.listdir(mask_source_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

            for mask_file in mask_files:
                mask_path = os.path.join(mask_source_dir, mask_file)

                try:
                    # 打开掩码图像（保持灰度）
                    mask = Image.open(mask_path)

                    # 确保是灰度模式
                    if mask.mode != 'L':
                        mask = mask.convert('L')

                    # 创建新的1920x1920黑色背景图像
                    new_mask = Image.new('L', (target_width, target_height), 0)

                    # 将原始掩码粘贴到中间
                    new_mask.paste(mask, (0, top_padding))

                    # 保存图像（使用PNG格式保持无损）
                    base_name = os.path.splitext(mask_file)[0]
                    target_path = os.path.join(target_seg_dir, f"{folder_name}_{base_name}.png")
                    new_mask.save(target_path, 'PNG')

                    print(f"  已处理掩码: {mask_file} -> {folder_name}_{base_name}.png")

                except Exception as e:
                    print(f"  处理掩码 {mask_file} 时出错: {e}")

    print("\n处理完成！")
    print(f"原图已保存到: {target_jpeg_dir}")
    print(f"掩码图已保存到: {target_seg_dir}")

    # 统计处理结果
    jpeg_count = len([f for f in os.listdir(target_jpeg_dir) if f.endswith('.jpg')])
    seg_count = len([f for f in os.listdir(target_seg_dir) if f.endswith('.png')])

    print(f"\n统计信息:")
    print(f"  原图数量: {jpeg_count}")
    print(f"  掩码图数量: {seg_count}")

    if jpeg_count != seg_count:
        print("警告: 原图和掩码图数量不一致！")


if __name__ == "__main__":
    # 检查源目录是否存在
    source_dir = r"D:\04_Media\Downloads\data"
    if not os.path.exists(source_dir):
        print(f"错误: 源目录 {source_dir} 不存在！")
        print("请确认路径是否正确。")
    else:
        process_images()