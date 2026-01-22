from PIL import Image
import os

def batch_jpg_to_png(input_dir, output_dir=None):
    """
    批量转换文件夹中的所有JPEG文件为PNG
    
    参数:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径，如果为None则创建"input_dir_png"
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = input_dir + '_png'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的JPEG扩展名
    jpeg_extensions = ['.jpg', '.jpeg', '.jpe', '.jfif']
    
    converted_count = 0
    
    for filename in os.listdir(input_dir):
        # 检查文件扩展名
        ext = os.path.splitext(filename)[1].lower()
        if ext in jpeg_extensions:
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                with Image.open(input_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, 'PNG')
                    converted_count += 1
                    print(f"✓ {filename} -> {output_filename}")
            except Exception as e:
                print(f"✗ {filename} 转换失败: {e}")
    
    print(f"\n批量转换完成！共转换 {converted_count} 个文件")
    print(f"输出目录: {output_dir}")

# 使用示例
batch_jpg_to_png('E:\\06_Temporary\\data_bridge\\masks', 'E:\\03_Learning\\MachineLearning\\unet-pytorch\\VOCdevkit\\VOC2007\\SegmentationClass')