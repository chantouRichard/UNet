import json
import os
import cv2
import numpy as np
import argparse
import tempfile
import shutil
from shapely.geometry import LineString, Polygon

def line_to_polygon(points, width):
    if len(points) < 2:
        return None
    line = LineString(points)
    # cap_style=2 (flat), join_style=2 (mitre)
    buffered = line.buffer(width / 2.0, cap_style=2, join_style=2)
    
    if isinstance(buffered, Polygon):
        coords = list(buffered.exterior.coords)
        return coords[:-1] if coords[0] == coords[-1] else coords
    return None

def process_and_visualize(image_path):
    base_dir = os.path.dirname(image_path)
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(base_dir, f"{file_name}.json")

    if not os.path.exists(json_path):
        print(f"❌ 错误: 没找到对应的 JSON 文件: {json_path}")
        return

    # 线条宽度映射表 (按你要求的 1->2, 2->3, 3->4, 4->5 更新)
    width_map = {"1": 2, "2": 3, "3": 4, "4": 5}

    temp_dir = tempfile.mkdtemp()
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("❌ 错误: 无法读取图片。")
            return
        
        # 创建一个和原图一样大的黑底掩码层 (用于计算面积)
        mask = np.zeros_like(img)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for shape in data['shapes']:
            label = str(shape['label'])
            shape_type = shape.get('shape_type', '').lower()
            points = shape['points']

            # 逻辑 A: 处理线段
            if shape_type in ['line', 'linestrip'] and label in width_map:
                w = width_map[label]
                poly_points = line_to_polygon(points, width=w)
                if poly_points:
                    pts = np.array(poly_points, np.int32)
                    cv2.fillPoly(mask, [pts], (0, 255, 0))

            # 逻辑 B: 处理 Label 10 的原生多边形
            elif label == "10" and shape_type == 'polygon':
                pts = np.array(points, np.int32)
                cv2.fillPoly(mask, [pts], (0, 255, 0))

        # --- 计算占比逻辑 ---
        # 将掩码转为单通道灰度图，非零部分即为标注区域
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        labeled_pixels = cv2.countNonZero(gray_mask)
        total_pixels = img.shape[0] * img.shape[1]
        percentage = (labeled_pixels / total_pixels) * 100

        print("-" * 30)
        print(f"📊 标注面积统计:")
        print(f"   总像素点: {total_pixels}")
        print(f"   标注像素: {labeled_pixels}")
        print(f"   掩码占比: {percentage:.4f}%")
        print("-" * 30)

        # 图像融合：0.6 是原图亮度，0.4 是掩码权重（稍微加深了一点掩码，方便观察）
        blended = cv2.addWeighted(img, 0.6, mask, 0.4, 0)

        # 保存并预览
        temp_img_path = os.path.join(temp_dir, "temp_preview.jpg")
        cv2.imwrite(temp_img_path, blended)
        
        print(f"✅ 处理完成。正在打开预览...")
        if os.name == 'nt':
            os.startfile(temp_img_path)
        else:
            opener = "open" if os.uname().sysname == "Darwin" else "xdg-open"
            os.system(f"{opener} {temp_img_path}")

        input("👉 预览已打开。按回车键关闭预览并清理临时文件...")

    finally:
        shutil.rmtree(temp_dir)
        print("🧹 临时文件已清理。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="图片路径")
    args = parser.parse_args()

    process_and_visualize(args.image)