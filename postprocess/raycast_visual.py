import os
import glob
import cv2
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm # 推荐使用 tqdm 显示进度条，如果没有请 pip install tqdm

# ==========================================
# 📂 1. 路径配置 (请根据你的实际项目路径微调)
# ==========================================
project_root = r"E:\03_Learning\MachineLearning\unet-pytorch"
pred_dir = os.path.join(project_root, "miou_out", "miou_unet_after", "detection-results")
gt_dir = os.path.join(project_root, "VOCdevkit", "VOC2007", "SegmentationClass")
output_dir = os.path.join(project_root, "healing_comparison_results")

# 如果输出文件夹不存在，则自动创建
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 📊 2. 评估初始化
# ==========================================
iou_before_list = []
iou_after_list = []
processed_count = 0

@njit
def zhang_suen_thinning_fast(image):
    # 确保输入是 np.uint8 类型，并且只有 0 和 1
    # 使用 copy 避免修改原图
    img = image.copy()
    rows, cols = img.shape
    
    # 预分配 marker 数组，避免在循环中重复申请内存
    marker = np.zeros((rows, cols), dtype=np.uint8)
    
    while True:
        has_changed = False
        marker[:, :] = 0
        
        # 第一步：子迭代1
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if img[i, j] == 0:
                    continue
                
                # 直接获取 8 邻域，避免创建 list 和调用函数
                p2 = img[i-1, j]
                p3 = img[i-1, j+1]
                p4 = img[i, j+1]
                p5 = img[i+1, j+1]
                p6 = img[i+1, j]
                p7 = img[i+1, j-1]
                p8 = img[i, j-1]
                p9 = img[i-1, j-1]
                
                Np = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if Np < 2 or Np > 6:
                    continue
                
                # 计算 0到1 的跳变次数 (Ap)
                Ap = 0
                if p2 == 0 and p3 == 1: Ap += 1
                if p3 == 0 and p4 == 1: Ap += 1
                if p4 == 0 and p5 == 1: Ap += 1
                if p5 == 0 and p6 == 1: Ap += 1
                if p6 == 0 and p7 == 1: Ap += 1
                if p7 == 0 and p8 == 1: Ap += 1
                if p8 == 0 and p9 == 1: Ap += 1
                if p9 == 0 and p2 == 1: Ap += 1
                
                if Ap == 1 and (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                    marker[i, j] = 1
                    has_changed = True
                    
        # 直接在原图上减去 marker
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if marker[i, j] == 1:
                    img[i, j] = 0
                    
        marker[:, :] = 0
        
        # 第二步：子迭代2
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if img[i, j] == 0:
                    continue
                
                p2 = img[i-1, j]
                p3 = img[i-1, j+1]
                p4 = img[i, j+1]
                p5 = img[i+1, j+1]
                p6 = img[i+1, j]
                p7 = img[i+1, j-1]
                p8 = img[i, j-1]
                p9 = img[i-1, j-1]
                
                Np = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if Np < 2 or Np > 6:
                    continue
                
                Ap = 0
                if p2 == 0 and p3 == 1: Ap += 1
                if p3 == 0 and p4 == 1: Ap += 1
                if p4 == 0 and p5 == 1: Ap += 1
                if p5 == 0 and p6 == 1: Ap += 1
                if p6 == 0 and p7 == 1: Ap += 1
                if p7 == 0 and p8 == 1: Ap += 1
                if p8 == 0 and p9 == 1: Ap += 1
                if p9 == 0 and p2 == 1: Ap += 1
                
                if Ap == 1 and (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                    marker[i, j] = 1
                    has_changed = True
                    
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if marker[i, j] == 1:
                    img[i, j] = 0
                    
        # 如果没有像素被删除，结束迭代
        if not has_changed:
            break
            
    return img

# ==========================================
# 辅助函数 1：提取单个连通域的端点
# ==========================================
def find_endpoints(skel_img):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skel_img, -1, kernel)
    endpoints_y, endpoints_x = np.where(filtered == 11)
    return list(zip(endpoints_y, endpoints_x))

# ==========================================
# 🟢 辅助函数 2：获取线段的【全局斜率】与【稳定锚点】
# ==========================================
def get_segment_features(comp_mask, dist_map, start_y, start_x, retreat_len=30):
    """
    计算全局斜率，并往回退缩 retreat_len 个像素寻找稳定锚点，避开末端倒钩。
    """
    ys, xs = np.where(comp_mask > 0)
    
    if len(xs) < 2:
        radius = dist_map[start_y, start_x]
        return (0.0, 0.0), max(1, int(np.round(radius * 2))), (start_y, start_x)

    pts = np.column_stack((xs, ys)).astype(np.float32)
    [vx, vy, cx, cy] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    global_vec = np.array([vy[0], vx[0]]) 
    
    centroid_y, centroid_x = np.mean(ys), np.mean(xs)
    outward_approx = np.array([start_y - centroid_y, start_x - centroid_x])
    if np.dot(global_vec, outward_approx) < 0:
        global_vec = -global_vec
        
    # 🟢 核心改动：顺着骨架往回走，寻找避开倒钩的“稳定锚点”
    curr_y, curr_x = start_y, start_x
    prev_y, prev_x = -1, -1
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for _ in range(retreat_len):
        found_next = False
        for dy, dx in offsets:
            ny, nx = curr_y + dy, curr_x + dx
            if 0 <= ny < comp_mask.shape[0] and 0 <= nx < comp_mask.shape[1]:
                if comp_mask[ny, nx] > 0 and (ny != prev_y or nx != prev_x):
                    prev_y, prev_x = curr_y, curr_x
                    curr_y, curr_x = ny, nx
                    found_next = True
                    break 
        if not found_next:
            break 
            
    anchor_y, anchor_x = curr_y, curr_x
    
    # 获取锚点处的粗细（倒钩尖端往往偏细，锚点处更代表主干真实粗细）
    radius = dist_map[anchor_y, anchor_x]
    thickness = max(1, int(np.round(radius * 2)))
    
    norm = np.hypot(global_vec[0], global_vec[1])
    final_vec = (0.0, 0.0) if norm == 0 else (global_vec[0] / norm, global_vec[1] / norm)
    
    # 返回：方向向量，粗细，锚点坐标
    return final_vec, thickness, (anchor_y, anchor_x)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line

# ==========================================
# 🟢 独家辅助函数：绘制渐变粗细的直线 (透视梯形法)
# ==========================================
def draw_gradient_thickness_line(img, pt1, pt2, thick1, thick2, color=255):
    """
    通过计算法向量构建多边形，绘制一头粗一头细的平滑过渡线。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 向量与距离
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    
    if length == 0:
        return
        
    # 计算归一化方向向量
    ux = dx / length
    uy = dy / length
    
    # 计算垂直法向量 (旋转90度)
    nx = -uy
    ny = ux
    
    # 计算两端的半径
    r1 = max(0.5, thick1 / 2.0)
    r2 = max(0.5, thick2 / 2.0)
    
    # 计算梯形的四个顶点
    p1 = (int(np.round(x1 + nx * r1)), int(np.round(y1 + ny * r1)))
    p2 = (int(np.round(x1 - nx * r1)), int(np.round(y1 - ny * r1)))
    p3 = (int(np.round(x2 - nx * r2)), int(np.round(y2 - ny * r2)))
    p4 = (int(np.round(x2 + nx * r2)), int(np.round(y2 + ny * r2)))
    
    # 用多边形填充梯形
    pts = np.array([p1, p2, p3, p4], dtype=np.int32)
    # OpenCV 接收的 pts 需要是一个 list
    if isinstance(color, tuple):
        cv2.fillPoly(img, [pts], color=color)
    else:
        cv2.fillPoly(img, [pts], color=int(color))
        
    # 在两端画圆润的帽子，防止梯形两端看起来像被刀切过一样平
    cv2.circle(img, (int(x1), int(y1)), int(np.round(r1)), color, -1)
    cv2.circle(img, (int(x2), int(y2)), int(np.round(r2)), color, -1)


# ==========================================
# 核心函数：中心双向延伸探测修复 (支持透视渐变)
# ==========================================
def ray_cast_midpoint_healing(binary_mask, skeleton, extend_ratio=0.1, hit_threshold=1):
    print("📍 正在生成距离变换图与划分连通域...")
    binary_255 = (binary_mask > 0).astype(np.uint8) * 255
    dist_map = cv2.distanceTransform(binary_255, cv2.DIST_L2, 5)
    
    skel_8u = (skeleton > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(skel_8u, connectivity=8)
    
    healed_mask = binary_255.copy()
    success_count = 0
    
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        
        segment_len = len(xs)
        if segment_len < 5:
            continue
            
        pts = np.column_stack((xs, ys)).astype(np.float32)
        [vx, vy, cx, cy] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        dir_x, dir_y = vx[0], vy[0]
        
        mid_x, mid_y = int(np.round(cx[0])), int(np.round(cy[0]))
        
        # 🟢 获取发射点(中点)的粗细
        source_thickness = max(1, int(np.round(dist_map[mid_y, mid_x] * 2)))
        
        ray_len = segment_len * (0.5 + extend_ratio)
        directions = [(dir_x, dir_y), (-dir_x, -dir_y)]
        
        for dx, dy in directions:
            end_x = int(np.round(mid_x + dx * ray_len))
            end_y = int(np.round(mid_y + dy * ray_len))
            
            rect = (0, 0, labels.shape[1], labels.shape[0])
            is_inside, pt1, pt2 = cv2.clipLine(rect, (mid_x, mid_y), (end_x, end_y))
            if not is_inside:
                continue
                
            rr, cc = line(pt1[1], pt1[0], pt2[1], pt2[0])
            hits = 0
            target_point = None
            
            for r, c in zip(rr, cc):
                if labels[r, c] == label_id:
                    continue 
                    
                encountered_label = labels[r, c]
                if encountered_label > 0 and encountered_label != label_id:
                    hits += 1
                    if hits >= hit_threshold:
                        target_point = (c, r) # 注意这里把 (x, y) 存下来
                        break 
                        
            if target_point is not None:
                hit_x, hit_y = target_point
                # 🟢 获取命中点(远端)的真实粗细
                target_thickness = max(1, int(np.round(dist_map[hit_y, hit_x] * 2)))
                
                # 🟢 使用透视梯形画笔！
                draw_gradient_thickness_line(healed_mask, 
                                             pt1=(mid_x, mid_y), 
                                             pt2=(hit_x, hit_y), 
                                             thick1=source_thickness, 
                                             thick2=target_thickness, 
                                             color=255)
                success_count += 1
                
    print(f"✅ 中心双向探测完毕！共触发了 {success_count} 次自适应透视延伸。")
    return healed_mask

# ==========================================
# 调试可视化函数：中点发射诊断图 (支持透视渐变)
# ==========================================
def visualize_midpoint_rays(binary_mask, skeleton, extend_ratio=0.1, hit_threshold=1):
    print("📍 开始中心射线追踪可视化分析...")
    binary_255 = (binary_mask > 0).astype(np.uint8) * 255
    dist_map = cv2.distanceTransform(binary_255, cv2.DIST_L2, 5)
    
    skel_8u = (skeleton > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(skel_8u, connectivity=8)
    
    h, w = binary_mask.shape
    debug_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    debug_canvas[binary_mask > 0] = [80, 80, 80]     
    debug_canvas[skeleton > 0] = [255, 255, 255]      
    
    success_count = 0
    failed_count = 0
    
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        
        if len(xs) < 5:
            continue
            
        pts = np.column_stack((xs, ys)).astype(np.float32)
        [vx, vy, cx, cy] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        dir_x, dir_y = vx[0], vy[0]
        
        mid_x, mid_y = int(np.round(cx[0])), int(np.round(cy[0]))
        source_thickness = max(1, int(np.round(dist_map[mid_y, mid_x] * 2)))
        
        cv2.circle(debug_canvas, (mid_x, mid_y), 5, (255, 0, 255), -1)
        
        ray_len = len(xs) * (0.5 + extend_ratio)
        directions = [(dir_x, dir_y), (-dir_x, -dir_y)]
        
        for dx, dy in directions:
            end_x = int(np.round(mid_x + dx * ray_len))
            end_y = int(np.round(mid_y + dy * ray_len))
            
            rect = (0, 0, w, h)
            is_inside, pt1, pt2 = cv2.clipLine(rect, (mid_x, mid_y), (end_x, end_y))
            if not is_inside:
                continue
                
            rr, cc = line(pt1[1], pt1[0], pt2[1], pt2[0])
            hits = 0
            target_point = None
            
            for r, c in zip(rr, cc):
                if labels[r, c] == label_id:
                    continue 
                    
                encountered_label = labels[r, c]
                if encountered_label > 0 and encountered_label != label_id:
                    hits += 1
                    if hits >= hit_threshold:
                        target_point = (c, r)
                        break
                        
            if target_point is not None:
                hit_x, hit_y = target_point
                target_thickness = max(1, int(np.round(dist_map[hit_y, hit_x] * 2)))
                
                # 🟢 命中：从中点画带透视效果的粗绿线
                draw_gradient_thickness_line(debug_canvas, 
                                             pt1=(mid_x, mid_y), 
                                             pt2=(hit_x, hit_y), 
                                             thick1=source_thickness, 
                                             thick2=target_thickness, 
                                             color=(0, 255, 0))
                success_count += 1
            else:
                # 🔴 未命中：继续画普通的红线代表射线探测路径
                cv2.line(debug_canvas, (mid_x, mid_y), (pt2[0], pt2[1]), 
                         color=(255, 0, 0), thickness=1)
                failed_count += 1
                
    print(f"✅ 诊断完毕！成功连上 {success_count} 处，射空了 {failed_count} 处。")
    return debug_canvas

def calculate_iou(pred_mask, gt_mask):
    """
    计算预测掩码和真实掩码的 IoU。
    输入可以是 0/255 的 uint8 图，也可以是 bool 数组。
    """
    # 统一转换为布尔型，方便进行逻辑运算
    pred_bool = (pred_mask > 0)
    
    # 注意：VOC 数据集中，边缘/忽略区域的像素值通常是 255
    # 如果你的 GT 只有目标(如1或255)和背景(0)，用 gt_mask > 0 即可。
    # 这里我们严格一点，排除了 255 这个 ignore label（如果存在的话）
    # 如果你的 GT 里白色(255)就是目标，请把 `& (gt_mask != 255)` 删掉
    gt_bool = (gt_mask > 0) & (gt_mask != 255) if np.max(gt_mask) == 255 and np.mean(gt_mask==255) < 0.1 else (gt_mask > 0)

    # 确保两者尺寸一致
    if pred_bool.shape != gt_bool.shape:
        raise ValueError(f"尺寸不匹配！预测图: {pred_bool.shape}, GT图: {gt_bool.shape}")

    # 计算交集 (Intersection) 和并集 (Union)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return intersection / union
# 获取所有预测结果的图片路径
pred_paths = glob.glob(os.path.join(pred_dir, "*.png"))

print(f"🚀 开始遍历评估，共找到 {len(pred_paths)} 张预测掩码...")

# ==========================================
# 🔄 3. 核心遍历循环
# ==========================================
for pred_path in tqdm(pred_paths, desc="Processing Images"):
    base_name = os.path.basename(pred_path)
    
    # 构建对应的 GT 路径
    gt_path = os.path.join(gt_dir, base_name)
    
    if not os.path.exists(gt_path):
        print(f"\n⚠️ 找不到对应的 GT 文件，跳过: {gt_path}")
        continue

    # ----------------------------------------
    # [步骤 A] 读取预测图与 GT 图
    # ----------------------------------------
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    # 确保预测图转为标准的 0 和 1 二值化格式
    binary = (pred_img > 0).astype(np.uint8)
    
    # ----------------------------------------
    # [步骤 B] 提取并优化骨架 (复用你提供的逻辑)
    # ----------------------------------------
    skeleton = zhang_suen_thinning_fast(binary)
    
    kernel_9x9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skeleton_closed = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel_9x9)
    skeleton = zhang_suen_thinning_fast(skeleton_closed)
    
    border_size = 5
    skeleton[:border_size, :] = 0  
    skeleton[-border_size:, :] = 0  
    skeleton[:, :border_size] = 0  
    skeleton[:, -border_size:] = 0  
    
    # ----------------------------------------
    # [步骤 C] 执行透视中心射线缝合
    # ----------------------------------------
    # extend_ratio=0.1 即前后各伸长 1/5
    healed_mask = ray_cast_midpoint_healing(binary, skeleton, extend_ratio=0.1, hit_threshold=1)
    
    # ----------------------------------------
    # [步骤 D] 计算 IoU
    # ----------------------------------------
    iou_b = calculate_iou(binary, gt_img)
    iou_a = calculate_iou(healed_mask, gt_img)
    
    iou_before_list.append(iou_b)
    iou_after_list.append(iou_a)
    processed_count += 1
    
    # ----------------------------------------
    # [步骤 E] 绘制并保存三联对比图
    # ----------------------------------------
    # 设置大尺寸画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Image: {base_name} | IoU Before: {iou_b:.4f} -> IoU After: {iou_a:.4f}", fontsize=16)
    
    # 图 1：原始预测
    axes[0].imshow(binary, cmap='gray')
    axes[0].set_title(f'Original Prediction\nIoU: {iou_b:.4f}', fontsize=14)
    axes[0].axis('off')
    
    # 图 2：修复后预测
    # 稍微做个彩色可视化，把修复的区域用红色标出来
    diff = healed_mask - binary
    show_healed = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    show_healed[binary > 0] = [200, 200, 200]  # 原有的线画成浅灰
    show_healed[diff > 0] = [255, 0, 0]        # 新缝合的线画成红色
    
    axes[1].imshow(show_healed)
    axes[1].set_title(f'Healed Prediction\nIoU: {iou_a:.4f}', fontsize=14)
    axes[1].axis('off')
    
    # 图 3：Ground Truth
    axes[2].imshow(gt_img, cmap='gray')
    axes[2].set_title('Ground Truth', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图片并关闭画布释放内存
    save_path = os.path.join(output_dir, f"cmp_{base_name}")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# ==========================================
# 📈 4. 输出最终统计结果
# ==========================================
if processed_count > 0:
    miou_before = np.mean(iou_before_list)
    miou_after = np.mean(iou_after_list)
    
    print("\n" + "="*40)
    print("🎉 全数据集评估完成！")
    print(f"📁 对比图已保存至: {output_dir}")
    print("-" * 40)
    print(f"📊 原始 mIoU (Before) : {miou_before:.4f}")
    print(f"📊 修复后 mIoU (After)  : {miou_after:.4f}")
    
    improvement = (miou_after - miou_before) * 100
    if improvement > 0:
        print(f"✨ 整体表现提升了 {improvement:.2f}%！算法非常有效！")
    else:
        print(f"⚠️ 整体表现下降了 {abs(improvement):.2f}%，可能需要调小 extend_ratio。")
    print("="*40)
else:
    print("❌ 没有找到任何可处理的图片，请检查路径。")