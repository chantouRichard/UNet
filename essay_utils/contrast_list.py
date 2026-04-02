import os
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础配置区
# ==========================================
# 你存放对比结果图的文件夹路径（即上一步代码的 SAVE_ROOT）
INPUT_DIR = 'comparison_results2'
# 最终拼接大图的保存路径
OUTPUT_FILE = 'final_comparison_grid.png'

# 想要展示的图片名称列表（行）
TARGET_NAMES = ['0122_001', '0037_011', '0008_004', '0167_001']

# 想要展示的列顺序（必须与你上一步生成的后缀完全对应）
COLUMN_SUFFIXES = [
    'Image', 
    'GT', 
    'BaseLine', 
    'DeepLabv3Plus', 
    'AttUnet', 
    'UnetPlusPlus', 
    'Ours'
]

# 显示在每列最下方的文本标签（用于论文展示的规范名称）
COLUMN_LABELS = [
    'Input Image', 
    'Ground Truth', 
    'Baseline', 
    'DeepLabV3+', 
    'Attention U-Net', 
    'UNet++', 
    'Ours'
]

# ==========================================
# 2. 绘图与拼接核心逻辑
# ==========================================
def create_comparison_grid():
    num_rows = len(TARGET_NAMES)
    num_cols = len(COLUMN_SUFFIXES)
    
    # 设置画布大小，这里按比例大致估算 (宽度，高度)
    # 你可以根据实际图片的横纵比微调这两个数值
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 3.5))
    
    # 如果只有一行或一列，axes 可能是一维的，这里将其强制转为二维数组便于遍历
    if num_rows == 1: axes = np.expand_dims(axes, axis=0)
    if num_cols == 1: axes = np.expand_dims(axes, axis=1)

    print("🖼️ 开始拼接图像网格...")

    for r, img_name in enumerate(TARGET_NAMES):
        for c, suffix in enumerate(COLUMN_SUFFIXES):
            ax = axes[r, c]
            
            # 构造图片路径
            img_path = os.path.join(INPUT_DIR, f"{img_name}_{suffix}.png")
            
            if os.path.exists(img_path):
                # 使用 cv2 读取并转换 BGR 为 RGB (matplotlib 需要 RGB)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                print(f"  [警告] 找不到图片: {img_path}，将留白。")
                # 如果找不到图片，画一个带有 "Missing" 字样的灰色空白块
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center', color='gray', fontsize=12)
                ax.set_facecolor('#eeeeee')
            
            # 去除每张小图的坐标轴和边框
            ax.axis('off')
            
            # 💡 核心逻辑：在最后一行（最下方）添加模型标签
            if r == num_rows - 1:
                # y=-0.15 表示在图片底部往下偏移一点的位置
                ax.text(0.5, -0.15, COLUMN_LABELS[c], 
                        transform=ax.transAxes, 
                        ha='center', va='top', 
                        fontsize=16, fontweight='bold', fontfamily='sans-serif')

    # 调整子图之间的间距，wspace为列间距，hspace为行间距
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.1)
    
    # 保存最终结果
    print(f"💾 正在保存高质量结果图至: {OUTPUT_FILE}")
    # bbox_inches='tight' 确保最下方的文字不会被截断，dpi=300 保证论文打印精度
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("✅ 拼接完成！")

if __name__ == '__main__':
    create_comparison_grid()