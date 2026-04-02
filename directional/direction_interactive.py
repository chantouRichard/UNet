import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 核心生成函数
# ==========================================
def generate_directional_field(mask_path):
    # 1. 读取单通道二值化 GT
    gt_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise FileNotFoundError(f"❌ 找不到掩码图片: {mask_path}")
        
    binary = (gt_img > 0) & (gt_img != 255)
    binary_8u = binary.astype(np.uint8)
    
    # 2. 距离变换
    dist = cv2.distanceTransform(binary_8u, cv2.DIST_L2, 5)
    
    # 3. Sobel 算子计算梯度
    dx = cv2.Sobel(dist, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(dist, cv2.CV_64F, 0, 1, ksize=3)
    
    # 4. 归一化梯度
    mag = np.hypot(dx, dy)
    mag[mag == 0] = 1e-5 
    nx = dx / mag
    ny = dy / mag
    
    # 5. 旋转 90 度，得到切向向量
    vx = -ny
    vy = nx
    
    # 过滤掉背景
    vx[~binary] = 0
    vy[~binary] = 0
    
    return binary, dist, vx, vy

# ==========================================
# 交互式可视化工具
# ==========================================
def interactive_direction_viewer(mask_path):
    # 1. 计算方向场
    binary, dist, vx, vy = generate_directional_field(mask_path)
    
    # 2. 尝试自动寻找对应的原图 (基于 VOC 数据集结构)
    img_path = mask_path.replace('SegmentationClass', 'JPEGImages').replace('.png', '.jpg')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title('Interactive Direction Field Viewer')
    
    # 3. 绘制背景
    if os.path.exists(img_path):
        print(f"✅ 找到对应原图: {img_path}")
        bg_img = cv2.imread(img_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        
        # 将掩码以半透明红色覆盖在原图上，方便你识别哪里有线段
        overlay = bg_img.copy()
        overlay[binary > 0] = [255, 0, 0] # 绳索区域标红
        bg_img = cv2.addWeighted(bg_img, 0.6, overlay, 0.4, 0)
        ax.imshow(bg_img)
    else:
        print(f"⚠️ 未找到原图 ({img_path})，将使用黑白掩码作为背景。")
        ax.imshow(binary, cmap='gray')

    ax.set_title("🖱️ Click on the cable to see its direction vector!\n(Check the console for detailed output)", fontsize=14)
    ax.axis('off')

    # 在图像左上角添加一个实时文本框
    info_text = ax.text(0.02, 0.98, 'Ready! Click on the red cables.', 
                        transform=ax.transAxes, color='cyan', fontsize=12, 
                        fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

    # ==========================================
    # 鼠标点击事件回调函数
    # ==========================================
    def onclick(event):
        # 确保点击在图像范围内
        if event.xdata is None or event.ydata is None:
            return
            
        # 获取点击像素的整数坐标
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        
        # 防止越界
        if x < 0 or x >= vx.shape[1] or y < 0 or y >= vx.shape[0]:
            return
            
        v_x, v_y = vx[y, x], vy[y, x]
        
        if binary[y, x]:
            # 计算角度 (用于直观显示)
            angle = np.degrees(np.arctan2(v_y, v_x))
            
            # 终端输出详细信息
            print(f"🎯 点击位置: (X={x:4d}, Y={y:4d}) | 向量: (vx={v_x:7.4f}, vy={v_y:7.4f}) | 角度: {angle:6.1f}°")
            
            # 更新图上的文本框
            info_text.set_text(f"Pos: ({x}, {y})\nVec: ({v_x:.2f}, {v_y:.2f})")
            
            # 在点击位置画一个醒目的黄色大点
            ax.scatter(x, y, color='yellow', s=50, zorder=4)
            
            # 🟢 在点击位置画出方向箭头 (Quiver)
            # 注意：在 imshow 中 Y 轴是朝下的，为了让箭头视觉上正确，设置 angles='xy' 和 scale_units='xy'
            ax.quiver(x, y, v_x, v_y, color='yellow', scale=0.03, scale_units='xy', angles='xy', width=0.005, zorder=5)
            
            # 刷新画布
            fig.canvas.draw()
        else:
            print(f"❌ 点击了背景 (X={x}, Y={y})，无方向向量。")
            info_text.set_text(f"Pos: ({x}, {y})\nBackground")
            fig.canvas.draw()

    # 绑定点击事件
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

# ==========================================
# 运行程序
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的 GT 掩码路径
    mask_file = r"VOCdevkit\VOC2007\SegmentationClass\0049_008.png"
    
    try:
        interactive_direction_viewer(mask_file)
    except FileNotFoundError as e:
        print(e)