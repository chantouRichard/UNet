import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def resize_images(source_dir, target_dir, target_size=512, is_mask=False, interpolation=cv2.INTER_LINEAR):
    """
    将目录中的所有图片缩放到指定尺寸

    参数:
        source_dir: 源图片目录
        target_dir: 目标目录
        target_size: 目标尺寸（正方形边长）
        is_mask: 是否为掩码/标签图片（保持灰度图）
        interpolation: 插值方法
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    # 支持的图片格式
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 获取所有图片文件
    source_path = Path(source_dir)
    img_files = [f for f in source_path.iterdir()
                 if f.suffix.lower() in img_extensions and f.is_file()]

    if not img_files:
        print(f"警告: {source_dir} 中没有找到图片文件")
        return 0

    print(f"正在处理 {source_dir} 中的 {len(img_files)} 张图片...")

    processed_count = 0
    for img_file in tqdm(img_files, desc=f"处理 {source_path.name}"):
        try:
            # 读取图片 - 如果是掩码，使用灰度模式读取
            if is_mask:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # 尝试用彩色读取再转换
                    img_color = cv2.imread(str(img_file))
                    if img_color is not None:
                        if len(img_color.shape) == 3:
                            # 如果是彩色图，转换为灰度
                            print(f"  警告: {img_file.name} 是彩色图，正在转换为灰度图...")
                            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                        else:
                            img = img_color
            else:
                img = cv2.imread(str(img_file))

            if img is None:
                print(f"  警告: 无法读取图片 {img_file.name}")
                continue

            # 获取原始尺寸
            if len(img.shape) == 2:  # 灰度图
                h, w = img.shape
                channels = 1
            else:  # 彩色图
                h, w = img.shape[:2]
                channels = img.shape[2]

            # 计算缩放比例并保持宽高比
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 缩放图片
            if len(img.shape) == 2:
                # 灰度图保持灰度
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
                # 创建512x512的黑色背景（单通道）
                target_img = np.zeros((target_size, target_size), dtype=np.uint8)
            else:
                # 彩色图
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
                # 创建512x512的黑色背景（3通道）
                target_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

            # 计算放置位置（居中）
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2

            # 将缩放后的图片放到黑色背景中间
            if len(img.shape) == 2:
                target_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
            else:
                target_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

            # 保存图片（保持原格式）
            target_path = Path(target_dir) / img_file.name

            # 对于掩码图片，确保保存为单通道
            if is_mask:
                # 检查是否为8位灰度图
                if target_img.dtype != np.uint8:
                    target_img = target_img.astype(np.uint8)
                cv2.imwrite(str(target_path), target_img)
                # 验证保存后的图片格式
                saved_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
                if saved_img is not None and len(saved_img.shape) != 2:
                    print(f"  错误: {img_file.name} 保存后仍不是灰度图!")
            else:
                cv2.imwrite(str(target_path), target_img)

            processed_count += 1

        except Exception as e:
            print(f"  处理图片 {img_file.name} 时出错: {e}")

    print(f"完成! 成功处理 {processed_count}/{len(img_files)} 张图片到 {target_dir}")
    return processed_count


def check_mask_format(mask_dir):
    """
    检查掩码图片格式，确保所有掩码都是灰度图
    """
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    mask_path = Path(mask_dir)

    if not mask_path.exists():
        print(f"警告: 目录 {mask_dir} 不存在")
        return False

    img_files = [f for f in mask_path.iterdir()
                 if f.suffix.lower() in img_extensions and f.is_file()]

    if not img_files:
        print(f"警告: {mask_dir} 中没有找到图片文件")
        return True

    all_gray = True
    for img_file in tqdm(img_files, desc="检查掩码格式"):
        # 以彩色模式读取检查通道数
        img = cv2.imread(str(img_file))
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] == 3:
                print(f"  发现彩色掩码: {img_file.name}")
                # 转换为灰度图
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(img_file), gray_img)
                print(f"    已转换为灰度图")
                all_gray = False

    return all_gray


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将图片批量缩放到指定尺寸')
    parser.add_argument('--source_jpeg', default='VOCdevkit/VOC2007-2-whole/JPEGImages',
                        help='源JPEG图片目录，默认: VOCdevkit/VOC2007-2-whole/JPEGImages')
    parser.add_argument('--source_seg', default='VOCdevkit/VOC2007-2-whole/SegmentationClass',
                        help='源分割掩码目录，默认: VOCdevkit/VOC2007-2-whole/SegmentationClass')
    parser.add_argument('--target_jpeg', default='VOCdevkit/VOC2007/JPEGImages',
                        help='目标JPEG图片目录，默认: VOCdevkit/VOC2007/JPEGImages')
    parser.add_argument('--target_seg', default='VOCdevkit/VOC2007/SegmentationClass',
                        help='目标分割掩码目录，默认: VOCdevkit/VOC2007/SegmentationClass')
    parser.add_argument('--size', type=int, default=1024,
                        help='目标尺寸，默认: 1024')
    parser.add_argument('--check_masks', action='store_true',
                        help='检查并转换掩码为灰度图')

    args = parser.parse_args()

    print("=" * 60)
    print("图片批量缩放工具")
    print(f"目标尺寸: {args.size}x{args.size}")
    print("=" * 60)

    # 如果启用了检查掩码选项，先检查和转换掩码
    if args.check_masks:
        print("\n[0] 检查掩码图片格式...")
        if os.path.exists(args.source_seg):
            check_mask_format(args.source_seg)
            print("掩码格式检查完成!")
        else:
            print(f"警告: 源掩码目录 {args.source_seg} 不存在")

    # 处理JPEGImages目录（彩色图片）
    print("\n[1] 处理原图 (JPEGImages)...")
    if os.path.exists(args.source_jpeg):
        # 对彩色图片使用高质量插值
        count_jpeg = resize_images(
            source_dir=args.source_jpeg,
            target_dir=args.target_jpeg,
            target_size=args.size,
            is_mask=False,  # 彩色图片
            interpolation=cv2.INTER_LANCZOS4  # 高质量插值
        )
    else:
        print(f"错误: 源目录 {args.source_jpeg} 不存在!")
        return

    # 处理SegmentationClass目录（灰度掩码）
    print("\n[2] 处理掩码图 (SegmentationClass)...")
    if os.path.exists(args.source_seg):
        # 对灰度掩码使用最近邻插值（避免引入新像素值）
        count_seg = resize_images(
            source_dir=args.source_seg,
            target_dir=args.target_seg,
            target_size=args.size,
            is_mask=True,  # 这是掩码/标签图片，保持灰度
            interpolation=cv2.INTER_NEAREST  # 最近邻插值，保持像素值不变
        )
    else:
        print(f"错误: 源目录 {args.source_seg} 不存在!")
        return

    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"原图处理: {count_jpeg} 张 -> {args.target_jpeg}")
    print(f"掩码处理: {count_seg} 张 -> {args.target_seg}")
    print("=" * 60)

    # 检查数量是否一致
    if count_jpeg != count_seg:
        print(f"警告: 原图和掩码图数量不一致 ({count_jpeg} vs {count_seg})")
        print("请检查是否有图片无法读取或格式错误")

    # 验证处理结果
    print("\n[3] 验证处理结果...")
    target_seg_path = Path(args.target_seg)
    if target_seg_path.exists():
        seg_files = list(target_seg_path.glob("*.png")) + list(target_seg_path.glob("*.jpg"))
        if seg_files:
            # 随机检查几张掩码图片
            import random
            test_files = random.sample(seg_files, min(3, len(seg_files)))
            print(f"随机检查 {len(test_files)} 张掩码图片:")
            for test_file in test_files:
                img = cv2.imread(str(test_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if len(img.shape) == 2:
                        print(f"  ✓ {test_file.name}: 灰度图, shape={img.shape}, 值范围=[{img.min()}, {img.max()}]")
                    else:
                        print(f"  ✗ {test_file.name}: 非灰度图, shape={img.shape}")
                else:
                    print(f"  ✗ {test_file.name}: 无法读取")

    # 使用说明
    print("\n" + "=" * 60)
    print("使用说明:")
    print(f"1. 训练前，将train.py中的图片路径修改为:")
    print(f"   - 图像路径: {args.target_jpeg}")
    print(f"   - 掩码路径: {args.target_seg}")
    print(f"2. 确保你的UNet网络输入层设置为 {args.size}x{args.size}")
    print("3. 建议检查掩码的像素值是否与类别数对应")
    print("\n常用命令:")
    print(f"python {__file__} --size 512  # 缩放到512x512")
    print(f"python {__file__} --check_masks  # 检查并转换掩码格式")
    print(f"python {__file__} --size 1024 --check_masks  # 完整处理")
    print("=" * 60)


if __name__ == "__main__":
    # 检查必要的库是否安装
    try:
        import cv2
        import numpy as np
        from tqdm import tqdm

        main()
    except ImportError as e:
        print("错误: 缺少必要的库")
        print(f"具体错误: {e}")
        print("\n请安装以下库:")
        print("pip install opencv-python numpy tqdm")
        print("或者: conda install opencv numpy tqdm")