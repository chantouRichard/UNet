import os
import json
import cv2
import numpy as np
from PIL import Image
import sys

def json_to_mask(json_path, image_path, mask_output_dir, image_output_dir):
    """
    从JSON文件中读取多边形标注并生成掩码
    """
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return False
        
        # 创建空白掩码（单通道，全黑）
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 处理JSON中的每个形状（假设是多边形标注）
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'polygon':
                # 提取多边形点
                points = np.array(shape['points'], dtype=np.int32)
                
                # 在掩码上绘制多边形（填充为白色）
                cv2.fillPoly(mask, [points], color=1)
        
        # 保存原始图像到输出目录
        image_filename = os.path.basename(image_path)
        image_output_path = os.path.join(image_output_dir, image_filename)
        cv2.imwrite(image_output_path, image)
        
        # 保存掩码图像
        mask_filename = image_filename.replace('.jpg', '.png')
        mask_output_path = os.path.join(mask_output_dir, mask_filename)
        cv2.imwrite(mask_output_path, mask)
        
        print(f"已处理: {image_filename}")
        return True
        
    except Exception as e:
        print(f"处理 {json_path} 时出错: {str(e)}")
        return False

def process_folder(input_dir, image_output_dir, mask_output_dir):
    """
    处理整个文件夹中的JSON和图像文件
    """
    # 确保输出目录存在
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            # 构建文件路径
            json_path = os.path.join(input_dir, filename)
            image_filename = filename.replace('.json', '.jpg')
            image_path = os.path.join(input_dir, image_filename)
            
            # 检查图像文件是否存在
            if os.path.exists(image_path):
                # 处理文件
                if json_to_mask(json_path, image_path, mask_output_dir, image_output_dir):
                    processed_count += 1
                else:
                    error_count += 1
            else:
                print(f"警告: 未找到图像文件 {image_filename}")
                error_count += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"失败: {error_count} 个文件")

if __name__ == "__main__":
    # 设置路径（根据你的实际情况修改）
    input_dir = r"E:\\06_Temporary\\bridge_poly"
    image_output_dir = r"E:\\06_Temporary\\data_bridge\\img"
    mask_output_dir = r"E:\\06_Temporary\\data_bridge\\masks"
    
    # 如果需要命令行参数，可以取消下面的注释
    # if len(sys.argv) != 4:
    #     print("使用方法: python generate_masks.py <输入文件夹> <图像输出文件夹> <掩码输出文件夹>")
    #     sys.exit(1)
    # 
    # input_dir = sys.argv[1]
    # image_output_dir = sys.argv[2]
    # mask_output_dir = sys.argv[3]
    
    print(f"输入目录: {input_dir}")
    print(f"图像输出目录: {image_output_dir}")
    print(f"掩码输出目录: {mask_output_dir}")
    print("开始处理...\n")
    
    process_folder(input_dir, image_output_dir, mask_output_dir)
    
    print(f"\n图像已保存到: {image_output_dir}")
    print(f"掩码已保存到: {mask_output_dir}")