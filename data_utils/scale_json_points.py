import os
import json
import base64
import cv2

def update_json_with_scaled_image(input_json_folder, scaled_image_folder, output_folder, scale_factor=4):
    """
    更新JSON文件：缩放坐标并更新imageData
    
    参数:
    input_json_folder: 原始JSON文件夹路径
    scaled_image_folder: 4倍放大后的图片文件夹路径
    output_folder: 输出JSON文件夹路径
    scale_factor: 缩放因子，默认为4
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_json_folder) if f.endswith('.json')]
    
    for json_file in json_files:
        try:
            # 1. 读取原始JSON文件
            json_path = os.path.join(input_json_folder, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 2. 构建对应的图片文件名（去除.json后缀，加上.jpg/.png）
            base_name = os.path.splitext(json_file)[0]
            
            # 查找对应的图片文件
            img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            img_path = None
            
            for ext in img_extensions:
                possible_img_path = os.path.join(scaled_image_folder, base_name + ext)
                if os.path.exists(possible_img_path):
                    img_path = possible_img_path
                    break
            
            if img_path is None:
                print(f"警告: 找不到 {base_name} 对应的图片文件，跳过")
                continue
            
            # 3. 读取4倍图片并转换为base64
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}，跳过")
                continue
            
            # 将图片编码为base64
            _, buffer = cv2.imencode(os.path.splitext(img_path)[1], img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 4. 更新JSON数据
            # 更新imageData
            data["imageData"] = img_base64
            
            # 更新imagePath（如果需要）
            data["imagePath"] = os.path.basename(img_path)
            
            # 更新imageHeight和imageWidth
            data["imageHeight"] = img.shape[0]
            data["imageWidth"] = img.shape[1]
            
            # 缩放points坐标（乘以4）
            if "shapes" in data:
                for shape in data["shapes"]:
                    if "points" in shape:
                        shape["points"] = [[x * scale_factor, y * scale_factor] 
                                          for x, y in shape["points"]]
            
            # 5. 保存更新后的JSON文件
            output_path = os.path.join(output_folder, json_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"已处理: {json_file} -> 图片尺寸: {img.shape[1]}x{img.shape[0]}")
            
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    
    print(f"\n处理完成！文件已保存到: {output_folder}")

# 使用示例
if __name__ == "__main__":
    input_json_folder = "E:\\06_Temporary\\bridge"  # 原始JSON文件夹
    scaled_image_folder = "E:\\06_Temporary\\data_bridge_new\\img"    # 4倍放大后的图片文件夹
    output_folder = "E:\\06_Temporary\\data_bridge_new\\img"       # 输出JSON文件夹
    
    update_json_with_scaled_image(input_json_folder, scaled_image_folder, output_folder, scale_factor=4)
