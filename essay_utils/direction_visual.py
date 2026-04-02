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
    
    # 1. 提取单像素骨架 (核心！)
    skeleton = skeletonize(binary)
    
    # 2. 计算距离变换和方向场
    dist = cv2.distanceTransform(binary_8u, cv2.DIST_L2, 5)
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
    
    return binary, skeleton, vx, vy


def visualize_and_save(img_path, mask_path, save_path="direction_field_vis.png"):
    # ===== 1. 读取原图 (图a) =====
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"找不到原图: {img_path}")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    orig_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if orig_mask is None:
        raise FileNotFoundError(f"找不到标签图片: {mask_path}")
    orig_mask[orig_mask > 0] = 255 # 二值化标签，方便可视化

    # ===== 2. 获取标签、骨架与方向场 =====
    binary, skeleton, vx, vy = generate_directional_field(mask_path)
    h, w = binary.shape

    # ===== 3. 制作 Binary Mask & Skeleton (图b) =====
    # 背景黑，拉索白，骨架线纯红
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask_vis[binary] = [255, 255, 255]
    mask_vis[skeleton] = [255, 0, 0]

    # ===== 4. 自动寻找最佳放大区域 =====
    y_coords, x_coords = np.where(skeleton > 0)
    if len(y_coords) > 0:
        cy, cx = int(np.median(y_coords)), int(np.median(x_coords))
    else:
        cy, cx = h//2, w//2

    half_w = 40  # 放大框的半宽 (可根据你的拉索粗细适当调大调小)
    y1, y2 = max(0, cy - half_w), min(h, cy + half_w)
    x1, x2 = max(0, cx - half_w), min(w, cx + half_w)

    # 在图a和图b上画黄色虚线框，提示审稿人这里被放大了
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.rectangle(orig_mask, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # ===== 5. 制作 Directional Field (图c) =====
    zoom_mask = binary[y1:y2, x1:x2]
    zoom_skel = skeleton[y1:y2, x1:x2]
    
    # 放大部分的底图：背景黑，拉索灰暗，骨架亮红
    zoom_vis = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    zoom_vis[zoom_mask] = [100, 100, 100]  # 拉索涂暗灰，为了让彩色箭头更明显
    zoom_vis[zoom_skel] = [255, 0, 0]      # 红色骨架

    # 准备箭头数据
    Y, X = np.mgrid[y1:y2, x1:x2]
    U = vx[y1:y2, x1:x2]
    V = vy[y1:y2, x1:x2]
    angles = np.arctan2(V, U) # 计算角度，用于上色

    # 只在拉索内部画箭头，每隔 step 个像素画一个
    step = 2 
    valid_mask = zoom_mask[::step, ::step]
    X_q = X[::step, ::step][valid_mask]
    Y_q = Y[::step, ::step][valid_mask]
    U_q = U[::step, ::step][valid_mask]
    V_q = V[::step, ::step][valid_mask]
    C_q = angles[::step, ::step][valid_mask] # 颜色值

    # ===== 6. 绘图与排版 =====
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # (a)
    axes[0].imshow(orig_img)
    axes[0].set_title("(a) Input Image", fontsize=15, pad=10)
    axes[0].axis('off')

    # (b)
    axes[1].imshow(mask_vis)
    axes[1].set_title("(b) Mask & Skeleton", fontsize=15, pad=10)
    axes[1].axis('off')

    # (c)
    axes[2].imshow(zoom_vis)
    # cmap='hsv' 保证不同方向的箭头颜色鲜艳不同
    axes[2].quiver(
        X_q - x1, Y_q - y1, U_q, V_q, C_q,
        cmap='hsv',
        scale=18,       # 箭头长度比例，数值越小箭头越长
        width=0.005,    # 箭头粗细
        headwidth=4
    )
    axes[2].set_title("(c) Directional Field", fontsize=15, pad=10)
    axes[2].axis('off')
    
    # (d) 放大区域的 GT 标签，供审稿人对照
    axes[3].imshow(orig_mask, cmap='gray')
    axes[3].set_title("(d) Zoomed GT Mask", fontsize=15, pad=10)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ 完美！论文配图已保存至: {save_path}")
    plt.close()


# ==============================
# 修改这里：填入你想要展示的一组图片的路径
# ==============================
img_file = r"VOCdevkit\VOC2007\JPEGImages\0001_000.jpg"        # 原图路径
gt_file = r"VOCdevkit\VOC2007\SegmentationClass\0001_000.png" # 标签路径

visualize_and_save(img_file, gt_file, "Figure3_Directional_Field.png")