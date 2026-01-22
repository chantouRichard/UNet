#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像低像素块检测与清理脚本
功能：检测masks目录中的图像，若非0像素低于阈值，则删除所有子目录中的同名文件
"""

import os
from pathlib import Path
import cv2
import numpy as np

# ==================== 配置区域（请根据需要修改） ====================
BASE_DIR = Path(r"data\bridge_filter_cropped")  # 主目录路径
SUBDIRS = ["img", "masks", "previews", "pure_color_masks"]  # 子目录列表
TARGET_DIR = "masks"  # 需要检测的子目录名
THRESHOLD_PERCENT = 5.0  # 非0像素百分比阈值（如1表示非0像素需>1%）
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}  # 支持的图片格式
DRY_RUN = False  # 设为True时只检测不删除（安全模式）


# ==================================================================

def calculate_nonzero_percentage(image_path):
    """
    计算图像中非0像素的百分比

    Args:
        image_path: 图像文件路径

    Returns:
        float: 非0像素百分比，读取失败返回-1
    """
    try:
        # 以灰度模式读取图像
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ❌ 无法读取图像: {image_path}")
            return -1

        # 计算总像素数和非0像素数
        total_pixels = img.size
        nonzero_pixels = np.count_nonzero(img)

        # 计算百分比
        percentage = (nonzero_pixels / total_pixels) * 100
        return percentage

    except Exception as e:
        print(f"  ❌ 处理图像时出错 {image_path}: {e}")
        return -1


def delete_files_safely(file_paths):
    """
    安全删除多个文件

    Args:
        file_paths: 待删除的文件路径列表
    """
    for file_path in file_paths:
        try:
            if file_path.exists():
                if not DRY_RUN:
                    file_path.unlink()
                # print(f"  {'⏳ 标记删除' if DRY_RUN else '🗑️ 已删除'}: {file_path.name}")
            else:
                print(f"  ⚠️  文件不存在: {file_path}")
        except Exception as e:
            print(f"  ❌ 删除失败 {file_path}: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("图像低像素块检测与清理工具")
    print("=" * 60)
    print(f"📁 主目录: {BASE_DIR}")
    print(f"🔍 检测目录: {TARGET_DIR}")
    print(f"📊 阈值: {THRESHOLD_PERCENT}%")
    print(f"🛡️  安全模式: {'是' if DRY_RUN else '否'}")
    print("=" * 60)

    # 检查基础目录
    if not BASE_DIR.exists():
        print(f"❌ 错误: 主目录不存在: {BASE_DIR}")
        return

    # 检查目标检测目录
    target_path = BASE_DIR / TARGET_DIR
    if not target_path.exists():
        print(f"❌ 错误: 检测目录不存在: {target_path}")
        return

    # 统计信息
    total_files = 0
    deleted_files = 0
    error_files = 0

    # 遍历目标目录中的所有图像文件
    print(f"\n开始扫描目录: {target_path}\n")

    for img_file in target_path.iterdir():
        # 只处理支持的图像格式
        if img_file.suffix.lower() not in SUPPORTED_EXTS:
            continue

        total_files += 1
        print(f"📷 处理: {img_file.name}")

        # 计算非0像素百分比
        nonzero_percent = calculate_nonzero_percentage(img_file)

        if nonzero_percent < 0:
            error_files += 1
            continue

        # print(f"   非0像素: {nonzero_percent:.2f}%")

        # 判断是否低于阈值
        if nonzero_percent < THRESHOLD_PERCENT:
            # print(f"   ⚠️  检测到非0像素低于阈值 ({THRESHOLD_PERCENT}%)")

            # 收集所有子目录中同名文件的路径
            files_to_delete = []
            for subdir in SUBDIRS:
                file_to_check = BASE_DIR / subdir / img_file.name
                files_to_delete.append(file_to_check)

            # print(f"   准备删除 {len(files_to_delete)} 个同名文件...")
            delete_files_safely(files_to_delete)
            deleted_files += 1

        # print("-" * 60)

    # 输出总结
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"📊 总计检查: {total_files} 个文件")
    print(f"🗑️  删除文件: {deleted_files} 组")
    print(f"❌ 错误文件: {error_files} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()