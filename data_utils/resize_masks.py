import os
import cv2
import numpy as np

def resize_masks(source_dir, target_dir, scale_factor=4):
    """
    将源文件夹中的mask图像放大4倍并保存到目标文件夹
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        # 只处理图像文件（支持常见格式）
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建完整路径
            source_path = os.path.join(source_dir, filename)
            
            # 读取图像（灰度模式）
            mask = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # 获取原始尺寸
                height, width = mask.shape
                
                # 计算新尺寸（4倍）
                new_width = width * scale_factor
                new_height = height * scale_factor
                
                # 使用最近邻插值（适合mask图像）
                resized_mask = cv2.resize(mask, (new_width, new_height), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # 保存到目标文件夹（保持原文件名）
                target_path = os.path.join(target_dir, filename)
                cv2.imwrite(target_path, resized_mask)
                
                print(f"已处理: {filename} ({width}x{height} -> {new_width}x{new_height})")
            else:
                print(f"错误: 无法读取图像 {filename}")

if __name__ == "__main__":
    # 设置路径（根据你的实际情况修改）
    source_dir = r"E:\\06_Temporary\\data_bridge_2\\masks"  # 源mask文件夹
    target_dir = r"E:\\06_Temporary\\data_bridge_new\\masks"  # 目标文件夹
    
    print(f"源文件夹: {source_dir}")
    print(f"目标文件夹: {target_dir}")
    print("开始处理...\n")
    
    resize_masks(source_dir, target_dir, scale_factor=4)
    
    print(f"\n处理完成！所有mask图像已放大4倍并保存到: {target_dir}")