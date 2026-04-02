import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_directional_field(mask_path):
    # 1. 读取单通道二值化 GT (只取前景区域)
    gt_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise FileNotFoundError(f"找不到图片: {mask_path}")
        
    # 忽略 255 的边界线，只保留目标 (假设目标是 1，按需调整)
    binary = (gt_img > 0) & (gt_img != 255)
    binary_8u = binary.astype(np.uint8)
    
    # 2. 计算距离变换 (Distance Transform)
    dist = cv2.distanceTransform(binary_8u, cv2.DIST_L2, 5)
    
    # 3. 使用 Sobel 算子计算 X 和 Y 方向的梯度
    # 使用 CV_64F 防止截断负数梯度
    dx = cv2.Sobel(dist, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(dist, cv2.CV_64F, 0, 1, ksize=3)
    
    # 4. 归一化梯度，得到单位“向心向量” (法向量)
    mag = np.hypot(dx, dy)
    mag[mag == 0] = 1e-5 # 防止除以零
    nx = dx / mag
    ny = dy / mag
    
    # 5. 🟢 核心魔法：旋转 90 度，得到顺着缆索走向的“切向向量”
    vx = -ny
    vy = nx
    
    # 过滤掉背景，只保留前景区域的向量
    vx[~binary] = 0
    vy[~binary] = 0
    
    return binary, dist, vx, vy

# ==========================================
# 可视化展示
# ==========================================
# 替换为你的 GT 路径
gt_file = r"VOCdevkit\VOC2007\SegmentationClass\0001_000.png"

try:
    binary, dist, vx, vy = generate_directional_field(gt_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 图 1：原始距离变换图
    axes[0].imshow(dist, cmap='jet')
    axes[0].set_title('Step 1: Distance Transform (Ridge)', fontsize=14)
    axes[0].axis('off')
    
    # 图 2：向量场 RGB 伪彩色映射
    # 将 (vx, vy) 映射到色相(Hue)和饱和度(Sat)，直观显示方向
    angle = (np.arctan2(vy, vx) * 180 / np.pi + 180) / 2 # 0~180
    hsv = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (binary * 255).astype(np.uint8)
    rgb_field = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    axes[1].imshow(rgb_field)
    axes[1].set_title('Step 2: Directional Field (Color Mapped)', fontsize=14)
    axes[1].axis('off')
    
    # 图 3：局部矢量箭头 (Quiver Plot)
    # 为了避免箭头密密麻麻看不清，我们每隔 5 个像素采样一次，并且截取一个 100x100 的局部
    y_coords, x_coords = np.where(binary > 0)
    if len(y_coords) > 0:
        cy, cx = int(np.median(y_coords)), int(np.median(x_coords))
        half_w = 50
        slice_y = slice(max(0, cy-half_w), min(binary.shape[0], cy+half_w))
        slice_x = slice(max(0, cx-half_w), min(binary.shape[1], cx+half_w))
        
        step = 3 # 采样间隔
        Y, X = np.mgrid[slice_y, slice_x]
        U = vx[slice_y, slice_x]
        V = vy[slice_y, slice_x]
        
        axes[2].imshow(binary[slice_y, slice_x], cmap='gray', alpha=0.5)
        axes[2].quiver(X[::step, ::step] - slice_x.start, 
                       Y[::step, ::step] - slice_y.start, 
                       U[::step, ::step], V[::step, ::step], 
                       color='red', scale=15, width=0.005)
        axes[2].set_title('Step 3: Vector Arrows (Local Zoom)', fontsize=14)
        axes[2].axis('off')
        axes[2].invert_yaxis() # matplotlib y轴向下
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(e)