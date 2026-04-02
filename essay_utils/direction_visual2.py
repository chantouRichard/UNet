import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def generate_directional_field(mask_path):
    gt_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise FileNotFoundError(f"找不到标签图片: {mask_path}")
        
    # 处理标签，确保二值化
    binary = (gt_img > 0)
    binary_8u = binary.astype(np.uint8)
    
    # 1. 提取单像素骨架
    skeleton = skeletonize(binary)
    
    # 2. 计算距离变换 (这里我们把 dist 也 return 出去，用于可视化)
    dist = cv2.distanceTransform(binary_8u, cv2.DIST_L2, 5)
    
    # 3. 计算方向场
    dx = cv2.Sobel(dist, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(dist, cv2.CV_64F, 0, 1, ksize=3)
    
    mag = np.hypot(dx, dy)
    mag[mag == 0] = 1e-5
    nx = dx / mag
    ny = dy / mag
    
    # 旋转 90 度得到拉索切线方向
    vx = -ny
    vy = nx
    
    vx[~binary] = 0
    vy[~binary] = 0
    
    return binary, skeleton, dist, vx, vy


def visualize_and_save(mask_path, save_path="Pipeline_Steps_Vis.png"):
    # ===== 1. 获取所有中间变量 =====
    binary, skeleton, dist, vx, vy = generate_directional_field(mask_path)
    h, w = binary.shape

    # ===== 2. 自动寻找最佳放大区域 =====
    # 寻找骨架像素集中的区域作为中心点
    y_coords, x_coords = np.where(skeleton > 0)
    if len(y_coords) > 0:
        cy, cx = int(np.median(y_coords)), int(np.median(x_coords))
    else:
        cy, cx = h//2, w//2

    half_w = 40  # 放大框的半宽 (可调)
    y1, y2 = max(0, cy - half_w), min(h, cy + half_w)
    x1, x2 = max(0, cx - half_w), min(w, cx + half_w)
    
    zoom_h, zoom_w = y2 - y1, x2 - x1

    # ===== 3. 裁剪所有数据到放大区域 =====
    zoom_binary = binary[y1:y2, x1:x2]
    zoom_skel   = skeleton[y1:y2, x1:x2]
    zoom_dist   = dist[y1:y2, x1:x2]
    zoom_vx     = vx[y1:y2, x1:x2]
    zoom_vy     = vy[y1:y2, x1:x2]

    # ===== 4. 制作图 (a): Skeleton =====
    # 背景纯黑，拉索区域暗灰(提供上下文)，骨架纯白
    vis_skel = np.zeros((zoom_h, zoom_w, 3), dtype=np.uint8)
    vis_skel[zoom_binary] = [60, 60, 60] 
    vis_skel[zoom_skel] = [255, 255, 255]

    # ===== 5. 制作图 (b): Distance Transform =====
    # 将距离场归一化并转为彩色热力图 (类似 Jet 伪彩)
    dist_norm = cv2.normalize(zoom_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_dist_bgr = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    vis_dist_rgb = cv2.cvtColor(vis_dist_bgr, cv2.COLOR_BGR2RGB)
    # 把非拉索的背景涂黑，让热力图只在拉索内部发光
    vis_dist_rgb[~zoom_binary] = [0, 0, 0]

    # ===== 6. 制作图 (c): Directional Field =====
    vis_dir = np.zeros((zoom_h, zoom_w, 3), dtype=np.uint8)
    vis_dir[zoom_binary] = [50, 50, 50]  # 背景涂暗，突出彩色箭头
    # 如果想保留红色的骨架线作为参考，可以取消下面这行的注释
    # vis_dir[zoom_skel] = [255, 0, 0]

    # 准备局部坐标系下的箭头数据
    Y, X = np.mgrid[0:zoom_h, 0:zoom_w]
    angles = np.arctan2(zoom_vy, zoom_vx) # 计算角度用于彩色映射

    # 每隔 step 个像素画一个箭头
    step = 2 
    valid_mask = zoom_binary[::step, ::step]
    X_q = X[::step, ::step][valid_mask]
    Y_q = Y[::step, ::step][valid_mask]
    U_q = zoom_vx[::step, ::step][valid_mask]
    V_q = zoom_vy[::step, ::step][valid_mask]
    C_q = angles[::step, ::step][valid_mask] 

    # ===== 7. 绘图与排版 =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Skeleton
    axes[0].imshow(vis_skel)
    axes[0].set_title("(a) Skeletonization", fontsize=16, pad=10, fontweight='bold')
    axes[0].axis('off')

    # (b) Distance Transform
    axes[1].imshow(vis_dist_rgb)
    axes[1].set_title("(b) Distance Transform", fontsize=16, pad=10, fontweight='bold')
    axes[1].axis('off')

    # (c) Directional Field
    axes[2].imshow(vis_dir)
    axes[2].quiver(
        X_q, Y_q, U_q, V_q, C_q,
        cmap='hsv',
        scale=18,       # 箭头长度比例
        width=0.005,    # 箭头粗细
        headwidth=4
    )
    axes[2].set_title("(c) Directional Field", fontsize=16, pad=10, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ 完美！架构图素材已保存至: {save_path}")
    plt.close()


# ==============================
# 填入标签路径 (原图在这里用不到了)
gt_file = r"VOCdevkit\VOC2007\SegmentationClass\0001_000.png" # 标签路径

visualize_and_save(gt_file, "Architecture_Pipeline_Stamps.png")