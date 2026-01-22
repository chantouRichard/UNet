import os
import shutil
import sys

def copy_json_and_jpg(source_dir, target_dir):
    """
    将源文件夹中的json文件及同名jpg文件复制到目标文件夹
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        # 只处理json文件
        if filename.endswith('.json'):
            # 构建json文件的完整路径
            json_path = os.path.join(source_dir, filename)
            
            # 构建同名jpg文件的完整路径
            jpg_filename = filename.replace('.json', '.jpg')
            jpg_path = os.path.join(source_dir, jpg_filename)
            
            # 检查jpg文件是否存在
            if os.path.exists(jpg_path):
                # 复制json文件
                shutil.copy2(json_path, target_dir)
                # 复制jpg文件
                shutil.copy2(jpg_path, target_dir)
                print(f"已复制: {filename} 和 {jpg_filename}")
            else:
                print(f"警告: 未找到与 {filename} 对应的jpg文件")

if __name__ == "__main__":
    # 从命令行参数获取源文件夹和目标文件夹路径
    if len(sys.argv) != 3:
        print("使用方法: python copy_files.py <源文件夹路径> <目标文件夹路径>")
        print("示例: python copy_files.py ./source ./target")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    
    copy_json_and_jpg(source_dir, target_dir)
    print("文件复制完成！")