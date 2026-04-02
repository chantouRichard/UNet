import os
import shutil

# 1. 定义路径
base_path = 'VOCdevkit/VOC2007'
txt_file = os.path.join(base_path, 'ImageSets/Segmentation/test.txt')
src_img_dir = os.path.join(base_path, 'JPEGImages')
src_mask_dir = os.path.join(base_path, 'SegmentationClass')

dst_img_dir = 'testSet/img'
dst_mask_dir = 'testSet/mask'

# 2. 创建目标文件夹（如果不存在）
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_mask_dir, exist_ok=True)

def copy_voc_files():
    # 3. 读取 test.txt
    if not os.path.exists(txt_file):
        print(f"错误: 找不到文件 {txt_file}")
        return

    with open(txt_file, 'r') as f:
        # 去掉每行末尾的换行符，并过滤空行
        image_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"开始处理，共发现 {len(image_names)} 个文件。")

    count = 0
    for name in image_names:
        # 拼接原图和掩码的文件名
        img_name = f"{name}.jpg"
        mask_name = f"{name}.png"

        # 定义完整的源路径和目标路径
        src_img = os.path.join(src_img_dir, img_name)
        src_mask = os.path.join(src_mask_dir, mask_name)
        
        dst_img = os.path.join(dst_img_dir, img_name)
        dst_mask = os.path.join(dst_mask_dir, mask_name)

        # 4. 执行复制操作
        # 复制原图
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        else:
            print(f"警告: 找不到原图 {src_img}")

        # 复制掩码
        if os.path.exists(src_mask):
            shutil.copy(src_mask, dst_mask)
        else:
            print(f"警告: 找不到掩码 {src_mask}")
        
        count += 1
        if count % 100 == 0:
            print(f"已处理 {count} 张图片...")

    print(f"任务完成！文件已存放到 {dst_img_dir} 和 {dst_mask_dir}")

if __name__ == "__main__":
    copy_voc_files()