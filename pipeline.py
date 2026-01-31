import os
import shutil
import tempfile
import argparse
import json

# 导入所有工具模块
from data_utils.test_labelme import convert_line_to_polygon_in_json  # 你的转换逻辑
from data_utils.generate_masks import process_folder
from data_utils.data_augmentor import SegmentationAugmentor
from data_utils.rename import rename_files_to_numbers
from data_utils.reshapeImg import main as reshape_images_in_folder
from data_utils.png2jpg import convert_png_to_jpg
from voc_annotation import main as run_voc_annotation

def run_pipeline(input_labelme_dir, project_root):
    """
    input_labelme_dir: 原始 LabelMe 图片和 JSON 的文件夹
    project_root: 项目根目录
    """
    voc_root = os.path.join(project_root, "VOCdevkit", "VOC2007")
    final_img_dir = os.path.join(voc_root, "JPEGImages")
    final_mask_dir = os.path.join(voc_root, "SegmentationClass")
    final_ImageSets_dir = os.path.join(voc_root, "ImageSets", "Segmentation")

    with tempfile.TemporaryDirectory() as working_dir:
        print(f"🚀 开始全自动预处理流水线... 临时目录: {working_dir}")

        # --- Step 0: 线段转多边形 (预处理 JSON) ---
        # 准备一个专门放处理后 JSON 的临时文件夹
        temp_json_dir = os.path.join(working_dir, "processed_jsons")
        os.makedirs(temp_json_dir)
        
        print("0. 正在进行 LabelMe 线段转多边形预处理...")
        # 将原始目录的文件拷贝/转换到 temp_json_dir
        for file in os.listdir(input_labelme_dir):
            if file.endswith('.json'):
                in_p = os.path.join(input_labelme_dir, file)
                out_p = os.path.join(temp_json_dir, file)
                convert_line_to_polygon_in_json(in_p, out_p)
            elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 图片直接拷贝过去，方便 generate_masks 读取
                shutil.copy(os.path.join(input_labelme_dir, file), temp_json_dir)

        # --- Step 1: generate_masks ---
        temp_img_raw = os.path.join(working_dir, "img_raw")
        temp_mask_raw = os.path.join(working_dir, "mask_raw")
        os.makedirs(temp_img_raw); os.makedirs(temp_mask_raw)
        
        print("1. 正在从预处理后的 JSON 生成原始 Mask...")
        # 这里读取的是我们 Step 0 处理好的 temp_json_dir
        process_folder(temp_json_dir, temp_img_raw, temp_mask_raw)

        # --- Step 2: data_augumentor (固定 3 个版本: 原图/翻转/光照) ---
        print("2. 正在进行数据增强 (原图+翻转+光照)...")
        temp_aug_out = os.path.join(working_dir, "aug_result")
        augmentor = SegmentationAugmentor(temp_aug_out)
        augmentor.run(temp_img_raw, temp_mask_raw)

        # --- Step 3: rename ---
        # 重命名完成后，temp_aug_out 下面现在有 img/ 和 masks/ 两个文件夹
        print("3. 正在执行统一数字重命名...")
        rename_files_to_numbers(temp_aug_out)

        # --- Step 4: reshapeImg ---
        print("4. 正在执行图像裁剪与尺寸统一 (Reshape)...")
        # 创建一个专门存放裁剪后结果的目录
        temp_reshaped_out = os.path.join(working_dir, "reshaped_data")
        
        # 调用修改后的 main 函数
        # source_path: 包含 img/ 和 masks/ 的父目录
        # output_path: 裁剪后存入的新目录
        # auto_confirm=True: 跳过手动输入确认
        reshape_images_in_folder(
            source_path=temp_aug_out, 
            output_path=temp_reshaped_out, 
            auto_confirm=True
        )

        # 更新后续步骤引用的路径，现在指向裁剪后的结果
        img_final_prep = os.path.join(temp_reshaped_out, "img")
        mask_final_prep = os.path.join(temp_reshaped_out, "masks")

        # --- Step 5: png2jpg ---
        print("5. 正在将裁剪后的原图转换为 JPG 格式...")
        convert_png_to_jpg(img_final_prep)

        # --- Step 6: 部署到 VOC 目录 ---
        print("6. 正在清理并部署到 VOCdevkit...")
        if os.path.exists(voc_root):
            shutil.rmtree(voc_root)
        os.makedirs(final_img_dir, exist_ok=True)
        os.makedirs(final_mask_dir, exist_ok=True)
        os.makedirs(final_ImageSets_dir, exist_ok=True)

        # 从裁剪并转换后的路径拷贝文件
        for f in os.listdir(img_final_prep):
            if f.lower().endswith('.jpg'):
                shutil.copy(os.path.join(img_final_prep, f), final_img_dir)
        for f in os.listdir(mask_final_prep):
            shutil.copy(os.path.join(mask_final_prep, f), final_mask_dir)

        # --- Step 7: voc_annotation ---
        print("7. 正在生成训练列表...")
        old_cwd = os.getcwd()
        os.chdir(project_root)
        try:
            run_voc_annotation()
        finally:
            os.chdir(old_cwd)

    print("\n✅ 所有处理已自动化完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始数据路径")
    parser.add_argument("--project", default=".", help="项目根目录")
    args = parser.parse_args()

    run_pipeline(args.input, args.project)