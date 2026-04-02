import cv2
import matplotlib.pyplot as plt
import os

def create_paper_figure():
    # === 1. 定义图片路径 ===
    # 假设你的三张图在同级的 fig3 目录下
    base_dir = "essay_utils/fig3"
    mask_path = os.path.join(base_dir, "mask.png")
    skeleton_path = os.path.join(base_dir, "skeleton.png")
    raycast_path = os.path.join(base_dir, "raycast.png")

    # === 2. 读取并转换颜色通道 (BGR -> RGB) ===
    # 注意：如果某张图不存在，代码会直接报错提示你
    try:
        mask_img = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        skeleton_img = cv2.cvtColor(cv2.imread(skeleton_path), cv2.COLOR_BGR2RGB)
        raycast_img = cv2.cvtColor(cv2.imread(raycast_path), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"❌ 读取图片失败，请检查 fig3 目录下是否有这三张图！\n错误信息: {e}")
        return

    # === 3. 创建 1x3 的画板 ===
    # figsize=(15, 5) 比例非常适合放在论文中跨越单栏
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # === 4. 绘制并设置子图属性 ===
    # (a) Predicted Mask
    axes[0].imshow(mask_img)
    axes[0].set_title("(a) Predicted Mask", fontsize=16, pad=15)
    axes[0].axis('off')  # 论文里的这种图必须去掉坐标轴

    # (b) Extracted Skeleton
    axes[1].imshow(skeleton_img)
    axes[1].set_title("(b) Extracted Skeleton", fontsize=16, pad=15)
    axes[1].axis('off')

    # (c) Ray Casting Healing
    axes[2].imshow(raycast_img)
    axes[2].set_title("(c) Ray Casting Healing", fontsize=16, pad=15)
    axes[2].axis('off')

    # === 5. 调整布局并保存高清图 ===
    plt.tight_layout()
    save_path = "Figure4_Healing_Pipeline.png"
    # dpi=300 和 bbox_inches='tight' 是学术出版标准的保存参数
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"✅ 完美！后处理原理图已成功保存至: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    create_paper_figure()