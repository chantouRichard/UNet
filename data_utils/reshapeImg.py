#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掩码数据集裁剪工具
对数据集中的图像和掩码进行重叠裁剪，生成固定尺寸的方形patches
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys
from PIL import Image

# ==================== 配置区域 ====================
# 源数据集目录
SOURCE_DATASET_DIR = Path("VOCdevkit\\VOC2007-temp")

# 输出数据集目录
OUTPUT_DATASET_DIR = Path("VOCdevkit\\VOC2007")

# 裁剪参数
PATCH_SIZE = 1024  # 裁剪尺寸（方形）
OVERLAP = 0.2  # 重叠比例 (0-1之间)

# 子目录名称配置
SUBDIRS = {
    'img': 'img',
    'masks': 'masks',
    # 'previews': 'previews',
    # 'pure_color_masks': 'pure_color_masks'
}

# 扩展名映射
EXTENSION_MAP = {
    'img': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'masks': ['.png'],
    # 'previews': ['.jpg', '.png'],
    # 'pure_color_masks': ['.png']
}

# 是否保留原始文件
KEEP_ORIGINAL = True

# 是否显示详细日志
VERBOSE = True

# 是否显示进度条
SHOW_PROGRESS = True

# 默认裁剪区域（当mask全黑或过滤后无有效patch时使用）
# 格式: (left, top, right, bottom) - 方形区域
DEFAULT_CROP_BOX = (896, 0, 1920, 1024)


# =================================================


def generate_content_patches(image: Image.Image, mask: np.ndarray,
                             patch_size: int = 1024, overlap: float = 0.2) -> List[Tuple[Image.Image, np.ndarray]]:
    """
    为超宽超大内容生成多个重叠的方形patches
    仅保留包含前景像素的patch，过滤纯背景patch

    Args:
        image: PIL.Image对象
        mask: numpy数组 (H, W)
        patch_size: 裁剪尺寸（宽高相同）
        overlap: 重叠比例

    Returns:
        List[Tuple[Image.Image, np.ndarray]]: (裁剪图像, 裁剪mask)的列表
    """
    # 确保mask是numpy数组
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # 找到mask中非零区域的边界
    non_zero_indices = np.where(mask > 0)

    # 如果mask全黑，返回默认裁剪区域
    if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
        left, top, right, bottom = DEFAULT_CROP_BOX
        return [(
            image.crop((left, top, right, bottom)),
            mask[top:bottom, left:right]
        )]

    # 计算内容边界（包含上下左右）
    content_min_y = np.min(non_zero_indices[0])
    content_max_y = np.max(non_zero_indices[0])
    content_min_x = np.min(non_zero_indices[1])
    content_max_x = np.max(non_zero_indices[1])

    patches = []
    step = int(patch_size * (1 - overlap))  # 重叠步长

    # 垂直方向滑动窗口
    top = max(0, content_min_y)
    while top < content_max_y:
        bottom = min(top + patch_size, image.height)

        # 如果最后一块不足patch_size，向上对齐
        if bottom == image.height and (bottom - top) < patch_size:
            top = max(0, bottom - patch_size)

        # 水平方向滑动窗口
        left = max(0, content_min_x)
        while left < content_max_x:
            right = min(left + patch_size, image.width)

            # 如果最后一块不足patch_size，向左对齐
            if right == image.width and (right - left) < patch_size:
                left = max(0, right - patch_size)

            # 裁剪mask区域
            crop_mask = mask[top:bottom, left:right]

            # ✅ 关键修复：只保留包含前景像素的patch
            if np.any(crop_mask > 0):
                crop_img = image.crop((left, top, right, bottom))
                patches.append((crop_img, crop_mask))

            left += step

            # 如果下一个窗口会超出图像右边界，则停止
            if left + patch_size > image.width:
                # 检查是否还有未覆盖的内容
                if left < content_max_x:
                    # 最后再添加一个右对齐的patch
                    right = image.width
                    left = max(0, right - patch_size)

                    crop_mask = mask[top:bottom, left:right]
                    if np.any(crop_mask > 0):
                        crop_img = image.crop((left, top, right, bottom))
                        patches.append((crop_img, crop_mask))
                break

        top += step

        # 如果下一个窗口会超出图像下边界，则停止
        if top + patch_size > image.height:
            # 检查是否还有未覆盖的内容
            if top < content_max_y:
                # 最后再添加一个底部对齐的patch
                bottom = image.height
                top = max(0, bottom - patch_size)

                # 水平方向重新滑动
                left = max(0, content_min_x)
                while left < content_max_x:
                    right = min(left + patch_size, image.width)

                    if right == image.width and (right - left) < patch_size:
                        left = max(0, right - patch_size)

                    crop_mask = mask[top:bottom, left:right]
                    if np.any(crop_mask > 0):
                        crop_img = image.crop((left, top, right, bottom))
                        patches.append((crop_img, crop_mask))

                    left += step

                    if left + patch_size > image.width:
                        if left < content_max_x:
                            right = image.width
                            left = max(0, right - patch_size)

                            crop_mask = mask[top:bottom, left:right]
                            if np.any(crop_mask > 0):
                                crop_img = image.crop((left, top, right, bottom))
                                patches.append((crop_img, crop_mask))
                        break
            break

    # ✅ 如果没有有效的patches（异常情况），返回默认裁剪区域
    if not patches:
        print(f"警告: 未找到包含前景像素的patch，使用默认裁剪区域")
        left, top, right, bottom = DEFAULT_CROP_BOX
        patches = [(
            image.crop((left, top, right, bottom)),
            mask[top:bottom, left:right]
        )]

    return patches


def scan_dataset_directory(dataset_dir: Path) -> List[Dict[str, Path]]:
    """
    扫描数据集目录，返回匹配的文件组
    """
    if VERBOSE:
        print(f"\n{'=' * 60}")
        print(f"正在扫描数据集: {dataset_dir}")
        print(f"{'=' * 60}")

    # 检查各子目录是否存在
    subdir_paths = {}
    for key, subdir_name in SUBDIRS.items():
        subdir_path = dataset_dir / subdir_name
        if subdir_path.exists():
            subdir_paths[key] = subdir_path
        else:
            print(f"警告: 子目录不存在 {subdir_path}")

    if not subdir_paths:
        print(f"错误: 未找到任何有效子目录")
        return []

    # 获取所有文件的基础名称（数字序号）
    base_indices = set()

    # 从img目录获取序号列表
    img_dir = subdir_paths.get('img')
    if img_dir and img_dir.exists():
        for file_path in img_dir.iterdir():
            if file_path.is_file() and file_path.stem.isdigit():
                base_indices.add(int(file_path.stem))

    if VERBOSE:
        print(f"找到 {len(base_indices)} 个基础序号")

    # 为每个序号查找对应的文件
    file_groups = []
    for idx in sorted(base_indices):
        file_group = {'index': idx}
        all_files_exist = True

        for subdir_key, subdir_path in subdir_paths.items():
            extensions = EXTENSION_MAP.get(subdir_key, [])
            found = False

            for ext in extensions:
                # 尝试不同扩展名
                file_path = subdir_path / f"{idx:04d}{ext}"
                if file_path.exists():
                    file_group[subdir_key] = file_path
                    found = True
                    break

            if not found:
                if VERBOSE:
                    print(f"警告: 序号 {idx} 在 {subdir_path} 中未找到文件")
                all_files_exist = False
                break

        if all_files_exist:
            file_groups.append(file_group)

    if VERBOSE:
        print(f"成功匹配 {len(file_groups)} 组完整文件")

    return file_groups


def create_output_directories():
    """创建输出目录结构"""
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for subdir_name in SUBDIRS.values():
        output_subdir = OUTPUT_DATASET_DIR / subdir_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        if VERBOSE:
            print(f"创建目录: {output_subdir}")


def process_file_pair(img_path: Path, mask_path: Path, base_idx: int,
                      step: int, subdir_key: str):
    """
    处理单对文件并保存裁剪结果
    """
    # 加载图像和mask
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # 生成patches
    patches = generate_content_patches(
        img, mask_array,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP
    )

    # 保存每个patch
    saved_files = []
    for patch_idx, (patch_img, patch_mask) in enumerate(patches):
        # 构建输出文件名: 原始序号_patch序号
        output_basename = f"{base_idx:04d}_{patch_idx:03d}"

        # 确定输出扩展名（保持原扩展名）
        img_ext = img_path.suffix
        mask_ext = mask_path.suffix

        # 保存img patch
        output_img_path = OUTPUT_DATASET_DIR / SUBDIRS[subdir_key] / f"{output_basename}{img_ext}"
        patch_img.save(output_img_path)
        saved_files.append(output_img_path)

        # 保存mask patch
        # mask对应的子目录名
        mask_subdir_key = 'masks' if subdir_key == 'img' else 'pure_color_masks'
        output_mask_path = OUTPUT_DATASET_DIR / SUBDIRS[mask_subdir_key] / f"{output_basename}{mask_ext}"
        Image.fromarray(patch_mask).save(output_mask_path)
        saved_files.append(output_mask_path)

    return saved_files


def process_dataset():
    """处理整个数据集"""
    # 1. 创建输出目录
    if VERBOSE:
        print("\n创建输出目录结构...")
    create_output_directories()

    # 2. 扫描数据集
    file_groups = scan_dataset_directory(SOURCE_DATASET_DIR)

    if not file_groups:
        print("\n错误: 未找到任何有效的文件组")
        sys.exit(1)

    # 3. 处理每个文件组
    total_groups = len(file_groups)

    print(f"\n{'=' * 60}")
    print(f"开始处理 {total_groups} 个文件组...")
    print(f"补丁大小: {PATCH_SIZE}x{PATCH_SIZE}, 重叠: {OVERLAP * 100}%")
    print(f"{'=' * 60}")

    total_patches_created = 0
    failed_groups = 0

    # 处理进度显示
    if SHOW_PROGRESS:
        try:
            from tqdm import tqdm
            progress = tqdm(file_groups, desc="处理进度", unit="组")
        except ImportError:
            progress = file_groups
    else:
        progress = file_groups

    for file_group in progress:
        base_idx = file_group['index']

        try:
            # 处理第一对: img 和 masks
            if 'img' in file_group and 'masks' in file_group:
                saved_files = process_file_pair(
                    file_group['img'], file_group['masks'],
                    base_idx, 0, 'img'
                )
                total_patches_created += len(saved_files) // 2

            # 处理第二对: previews 和 pure_color_masks
            # if 'previews' in file_group and 'pure_color_masks' in file_group:
            #     saved_files = process_file_pair(
            #         file_group['previews'], file_group['pure_color_masks'],
            #         base_idx, 0, 'previews'
            #     )
            #     total_patches_created += len(saved_files) // 2

            # 如果不需要保留原文件，删除原文件
            if not KEEP_ORIGINAL:
                # for subdir_key in ['img', 'masks', 'previews', 'pure_color_masks']:
                for subdir_key in ['img', 'masks']:
                    if subdir_key in file_group:
                        file_group[subdir_key].unlink()

            if VERBOSE and not SHOW_PROGRESS:
                print(f"序号 {base_idx:04d}: 完成")

        except Exception as e:
            print(f"错误处理序号 {base_idx:04d}: {str(e)}")
            failed_groups += 1
            continue

    # 打印统计信息
    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"处理的文件组: {total_groups}")
    print(f"创建的补丁总数: {total_patches_created}")
    print(f"失败的组数: {failed_groups}")
    print(f"输出目录: {OUTPUT_DATASET_DIR}")
    print(f"文件操作模式: {'保留原文件' if KEEP_ORIGINAL else '移动到新目录'}")
    print(f"{'=' * 60}")


def main():
    """主函数"""
    print("掩码数据集裁剪工具")
    print("=" * 60)
    print(f"源数据集: {SOURCE_DATASET_DIR}")
    print(f"输出数据集: {OUTPUT_DATASET_DIR}")
    print(f"补丁大小: {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"重叠比例: {OVERLAP * 100}%")
    print(f"默认裁剪区域: {DEFAULT_CROP_BOX}")
    print(f"操作模式: {'保留原文件' if KEEP_ORIGINAL else '移动（处理后删除原文件）'}")
    print("=" * 60)

    # 确认执行
    response = input("\n是否开始裁剪? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("操作已取消")
        sys.exit(0)

    # 处理数据集
    process_dataset()


if __name__ == "__main__":
    # 检查必要的库
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}")
        print("请安装: pip install numpy Pillow")
        sys.exit(1)

    main()