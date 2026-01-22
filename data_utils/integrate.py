import os
import shutil
from pathlib import Path

def merge_folders_keep_structure(source_root, target_root):
    """
    合并所有数字文件夹，保持img/masks/pure_color_masks的目录结构
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    
    # 创建目标文件夹
    (target_root / 'img').mkdir(parents=True, exist_ok=True)
    (target_root / 'masks').mkdir(parents=True, exist_ok=True)
    (target_root / 'pure_color_masks').mkdir(parents=True, exist_ok=True)
    
    # 数字文件夹范围
    num_folders = ['3', '4', '5', '6', '7', '8']
    
    for num in num_folders:
        num_path = source_root / num
        
        if not num_path.exists():
            print(f"警告: 文件夹 {num_path} 不存在")
            continue
            
        print(f"正在处理文件夹: {num}")
        
        # 处理每个子文件夹
        for subfolder in ['img', 'masks', 'pure_color_masks']:
            source_subfolder = num_path / subfolder
            target_subfolder = target_root / subfolder
            
            if source_subfolder.exists():
                for item in source_subfolder.iterdir():
                    if item.is_file():
                        # 为避免文件名冲突，可以添加前缀
                        new_name = f"{num}_{item.name}"
                        shutil.copy2(item, target_subfolder / new_name)
                        print(f"  复制: {item.name} -> {new_name}")
                    elif item.is_dir():
                        # 如果有子目录，递归复制
                        shutil.copytree(item, target_subfolder / item.name, 
                                      dirs_exist_ok=True)
    
    print(f"\n合并完成！文件保存在: {target_root}")

# 使用示例
if __name__ == "__main__":
    source_directory = "D:\\04_Media\\Downloads\\data"  # 当前目录，或替换为你的源目录路径
    target_directory = "D:\\04_Media\\Downloads\\data_integrate"  # 合并后的目标目录
    
    merge_folders_keep_structure(source_directory, target_directory)