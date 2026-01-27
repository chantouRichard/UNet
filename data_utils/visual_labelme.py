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

    # 线条宽度映射表
    width_map = {"1": 2, "2": 4, "3": 6, "4": 8, "9": 18}

    temp_dir = tempfile.mkdtemp()
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("❌ 错误: 无法读取图片。")
            return
        
        # 创建一个和原图一样大的掩码层
        mask = np.zeros_like(img)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for shape in data['shapes']:
            label = str(shape['label'])
            shape_type = shape['shape_type']
            points = shape['points']

            # 情况 A: 需要转换宽度的线条 (label 1, 2, 3, 4, 9)
            if shape_type in ['line', 'linestrip'] and label in width_map:
                w = width_map[label]
                poly_points = line_to_polygon(points, width=w)
                if poly_points:
                    pts = np.array(poly_points, np.int32)
                    cv2.fillPoly(mask, [pts], (0, 255, 0)) # 绿色填充

            # 情况 B: label 为 10 的原生多边形 (直接绘制区域)
            elif shape_type == 'polygon' and label == "10":
                pts = np.array(points, np.int32)
                cv2.fillPoly(mask, [pts], (0, 255, 0)) # 绿色填充

        # 图像融合：1.0 是原图权重，0.5 是掩码透明度
        blended = cv2.addWeighted(img, 0.6, mask, 0.2, 0)

        # 保存并调用本地查看器
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