import json
import numpy as np
from shapely.geometry import LineString, Polygon
import os


def line_to_polygon(points, width=5):
    """
    将中心线点列转换为带宽度的多边形顶点列表
    :param points: [[x1,y1], [x2,y2], ...] 中心线坐标
    :param width: 绳索总宽度（像素），建议 3~10
    :return: 多边形顶点列表 [[x1,y1], [x2,y2], ..., [xn,yn]]
    """
    if len(points) < 2:
        return None

    line = LineString(points)
    # 使用 buffer 生成“带状”区域（单位：像素）
    buffered = line.buffer(width / 2.0, cap_style=2, join_style=2)  # cap=flat, join=mitre

    if isinstance(buffered, Polygon):
        # 提取 exterior 坐标（逆时针顺序）
        coords = list(buffered.exterior.coords)
        # 移除最后一个重复点（shapely 会闭合）
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        return coords
    else:
        # 可能是 MultiPolygon 等（复杂自交情况），暂不处理
        print("Warning: Failed to create polygon from line.")
        return points  # fallback（不推荐）


def convert_line_to_polygon_in_json(json_path, output_path=None, width=5):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_shapes = []
    for shape in data['shapes']:
        if shape['shape_type'] in ['line', 'linestrip']:
            print(f"Converting line '{shape['label']}' to polygon...")
            points = shape['points']
            poly_points = line_to_polygon(points, width=width)
            if poly_points:
                shape['points'] = poly_points
                shape['shape_type'] = 'polygon'
        new_shapes.append(shape)

    data['shapes'] = new_shapes

    out_path = output_path or json_path.replace('.json', '_poly.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_path}")


# ===== 批量处理目录 =====
def batch_convert(input_dir, output_dir=None, width=5):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace('.json', '.json'))
            convert_line_to_polygon_in_json(input_path, output_path, width=width)


# ===== 使用示例 =====
if __name__ == "__main__":
    # 单个文件
    # convert_line_to_polygon_in_json("example.json", width=6)

    # 批量处理
    batch_convert(
        input_dir="E:\\06_Temporary\\data_bridge_new\\img",  # 原始 LabelMe JSON 目录（含 line 标注）
        output_dir="E:\\06_Temporary\\bridge_poly_2",  # 输出目录
        width=2  # 绳索宽度（像素）
    )