import os
import shutil
from tqdm import tqdm

# --- 路径配置 ---
# 1. test.txt 文件的路径
txt_path = r'VOCdevkit\VOC2007\ImageSets\Segmentation\test.txt'
# 2. 原始图片所在的文件夹
src_dir = r'VOCdevkit\VOC2007\JPEGImages'
# 3. 目标文件夹
dst_dir = r'test_JPEG'

# 如果目标文件夹不存在，则创建
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    print(f"创建文件夹: {dst_dir}")

def copy_test_images():
    # 检查 txt 文件是否存在
    if not os.path.exists(txt_path):
        print(f"错误: 找不到文件 {txt_path}")
        return

    # 读取 txt 中的所有文件名
    with open(txt_path, 'r') as f:
        # strip() 去除换行符，且过滤空行
        file_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"开始复制图片，共计 {len(file_names)} 张...")

    count = 0
    for name in tqdm(file_names):
        # 拼接原始文件路径（假设都是 .jpg 格式）
        src_path = os.path.join(src_dir, name + ".jpg")
        dst_path = os.path.join(dst_dir, name + ".jpg")

        # 检查源文件是否存在
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            count += 1
        else:
            print(f"\n警告: 找不到图片 {src_path}")

    print(f"\n任务完成！成功复制 {count} 张图片到 {dst_dir}")

if __name__ == "__main__":
    copy_test_images()