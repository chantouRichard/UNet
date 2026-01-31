import json
import numpy as np
from shapely.geometry import LineString, Polygon
import os
from tqdm import tqdm
def line_to_polygon(points, width=5):
    """
    将中心线点列转换为带宽度的多边形顶点列表
    """
    if len(points) < 2:
        return None

    line = LineString(points)
    # cap_style=2 (flat), join_style=2 (mitre) 保持直角边缘
    buffered = line.buffer(width / 2.0, cap_style=2, join_style=2)

    if isinstance(buffered, Polygon):
        coords = list(buffered.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        return coords
    else:
        print("Warning: Failed to create polygon from line.")
        return points

def convert_line_to_polygon_in_json(json_path, output_path=None):
    # 定义 label 到 width 的映射关系
    # 格式: "label": width
    label_width_map = {
        "1": 2,
        "2": 3,
        "3": 4,
        "4": 5
    }

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_shapes = []
    for shape in data['shapes']:
        label = str(shape['label'])  # 转为字符串以防 label 是数字
        
        # 逻辑处理：如果是 label 10，直接保留原样 (保持多边形)
        if label == "10":
            new_shapes.append(shape)
            continue

        # 如果是 line 或 linestrip，根据映射表转换
        if shape['shape_type'] in ['line', 'linestrip']:
            # 获取对应的宽度，如果没有定义则默认设为 2
            target_width = label_width_map.get(label, 2)
            
            # tqdm.write(f"Converting line '{label}' to polygon with width {target_width}...")
            points = shape['points']
            poly_points = line_to_polygon(points, width=target_width)
            
            if poly_points:
                shape['points'] = poly_points
                shape['shape_type'] = 'polygon'
        
        new_shapes.append(shape)

    data['shapes'] = new_shapes

    out_path = output_path or json_path.replace('.json', '_poly.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ===== 批量处理目录 =====
def batch_convert(input_dir, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 修改部分：先获取所有 JSON 文件列表 ---
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # --- 修改部分：使用 tqdm 包装循环 ---
    for file in tqdm(json_files, desc="线段转多边形", unit="file"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        convert_line_to_polygon_in_json(input_path, output_path)

# ===== 使用示例 =====
if __name__ == "__main__":
    batch_convert(
        input_dir="E:\\06_Temporary\\data_bridge_new\\img",
        output_dir="E:\\06_Temporary\\bridge_poly_2"
    )