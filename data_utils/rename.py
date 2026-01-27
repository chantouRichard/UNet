import os
from pathlib import Path

# 配置参数 - 按你的实际情况修改
BASE_DIR = Path("VOCdevkit\\VOC2007-temp")  # 当前目录，或者改成你的数据集目录路径
FOLDERS = ["img", "masks", "pure_color_masks", "previews"]  # 要处理的文件夹列表

def rename_files_to_numbers():
    """
    将所有文件夹中的文件重命名为0001开始的四位数字
    按文件名排序，确保对应关系
    """
    print("开始重命名文件...")
    
    # 检查所有文件夹是否存在
    existing_folders = []
    for folder in FOLDERS:
        folder_path = BASE_DIR / folder
        if folder_path.exists():
            existing_folders.append(folder)
            print(f"找到文件夹: {folder}")
        else:
            print(f"警告: 文件夹 {folder} 不存在，跳过")
    
    if not existing_folders:
        print("错误: 没有找到任何文件夹！")
        return
    
    # 获取第一个文件夹的文件列表作为基准
    first_folder = existing_folders[0]
    first_folder_path = BASE_DIR / first_folder
    
    # 获取文件并按名称排序
    all_files = []
    for file_path in first_folder_path.iterdir():
        if file_path.is_file():
            all_files.append(file_path.name)
    
    # 按文件名排序（字母顺序）
    all_files.sort()
    
    print(f"\n在 {first_folder} 中找到 {len(all_files)} 个文件")
    
    # 为每个文件创建映射
    file_mapping = {}
    for idx, filename in enumerate(all_files, 1):
        # 生成新的数字名称，如0001、0002...
        new_name = f"{idx:04d}{Path(filename).suffix}"  # 保持原扩展名
        file_mapping[filename] = new_name
    
    # 显示一些示例映射
    print("\n前5个文件的重命名示例:")
    for i, (old, new) in enumerate(list(file_mapping.items())[:5]):
        print(f"  {old} -> {new}")
    
    # 重命名所有文件夹中的文件
    total_renamed = 0
    for folder in existing_folders:
        folder_path = BASE_DIR / folder
        print(f"\n处理文件夹: {folder}")
        
        # 获取该文件夹的文件并按相同顺序排序
        folder_files = []
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                folder_files.append(file_path.name)
        
        folder_files.sort()
        
        # 确保文件数量一致
        if len(folder_files) != len(all_files):
            print(f"警告: {folder} 有 {len(folder_files)} 个文件，但基准文件夹有 {len(all_files)} 个文件")
        
        # 重命名文件
        for idx, old_filename in enumerate(folder_files, 1):
            if idx <= len(file_mapping):
                # 获取对应的新名称
                old_name_key = all_files[idx-1]  # 使用基准文件夹的文件名作为键
                new_filename = file_mapping.get(old_name_key)
                
                if new_filename:
                    old_path = folder_path / old_filename
                    new_path = folder_path / new_filename
                    
                    # 检查是否已存在同名文件
                    if new_path.exists():
                        # 如果目标文件已存在，先删除（可能是之前的重命名残留）
                        new_path.unlink()
                    
                    try:
                        old_path.rename(new_path)
                        print(f"  ✓ {old_filename} -> {new_filename}")
                        total_renamed += 1
                    except Exception as e:
                        print(f"  ✗ 重命名失败 {old_filename}: {e}")
    
    print(f"\n{'='*50}")
    print(f"重命名完成！")
    print(f"共处理了 {len(existing_folders)} 个文件夹")
    print(f"总共重命名了 {total_renamed} 个文件")
    print(f"所有文件现在已按顺序命名为 0001, 0002, 0003...")

if __name__ == "__main__":
    rename_files_to_numbers()