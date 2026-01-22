import os
from PIL import Image
import sys

def convert_png_to_jpg(folder_path):
    """
    将指定文件夹中的所有PNG图片转换为JPG格式，并删除原PNG文件
    """
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 '{folder_path}' 不存在！")
            return
        
        # 统计转换情况
        converted_count = 0
        error_count = 0
        
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.png'):
                # 构建完整文件路径
                png_path = os.path.join(folder_path, filename)
                jpg_filename = os.path.splitext(filename)[0] + '.jpg'
                jpg_path = os.path.join(folder_path, jpg_filename)
                
                try:
                    # 打开PNG图片并转换为JPG
                    with Image.open(png_path) as img:
                        # 转换模式为RGB（去除alpha通道）
                        if img.mode in ('RGBA', 'LA', 'P'):
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        else:
                            rgb_img = img.convert('RGB')
                        
                        # 保存为JPG
                        rgb_img.save(jpg_path, 'JPEG', quality=95)
                    
                    # 删除原PNG文件
                    os.remove(png_path)
                    print(f"已转换: {filename} -> {jpg_filename}")
                    converted_count += 1
                    
                except Exception as e:
                    print(f"转换失败 {filename}: {str(e)}")
                    error_count += 1
        
        # 输出结果
        print(f"\n转换完成！")
        print(f"成功转换: {converted_count} 张图片")
        print(f"失败: {error_count} 张图片")
        
        if converted_count == 0:
            print("提示：未找到PNG图片文件")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    # 使用方法说明
    if len(sys.argv) != 2:
        print("用法: python png_to_jpg.py <文件夹路径>")
        print("示例: python png_to_jpg.py ./images")
        print("\n或者直接修改脚本中的文件夹路径变量")
        
        # 如果需要，可以直接在这里指定文件夹路径
        # folder = "C:/Users/YourName/Pictures"  # 取消注释并修改为你的路径
        # convert_png_to_jpg(folder)
    else:
        folder_path = sys.argv[1]
        convert_png_to_jpg(folder_path)