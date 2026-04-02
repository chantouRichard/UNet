import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt # 新增：用于绘图

# ==========================================
# 📂 1. 路径配置
# ==========================================
project_root = r"E:\03_Learning\MachineLearning\unet-pytorch"

gt_dir       = os.path.join(project_root, "VOCdevkit", "VOC2007", "SegmentationClass")
pred_base_dir= os.path.join(project_root, "miou_out", "miou_unet_100epoch", "detection-results") # Baseline
pred_dir_dir = os.path.join(project_root, "miou_out", "miou_unet_dir_cbam", "detection-results")  # 方向场创新模型

# 新增：保存可视化对比图的文件夹
out_visual_dir = os.path.join(project_root, "tradeoff_analysis_visuals")
os.makedirs(out_visual_dir, exist_ok=True)

# ==========================================
# 🧮 2. 评估指标计算与可视化函数
# ==========================================
def compute_basic_metrics(pred, gt, valid_mask):
    """同时计算 IoU, Precision, Recall"""
    p = pred[valid_mask]
    g = gt[valid_mask]
    
    TP = np.logical_and(p, g).sum()
    FP = np.logical_and(p, ~g).sum()
    FN = np.logical_and(~p, g).sum()

    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return iou, precision, recall

def compute_cldice(pred, gt, valid_mask):
    """计算 clDice"""
    pred_b = (pred > 0) & valid_mask
    gt_b   = (gt > 0) & valid_mask
    
    if pred_b.sum() == 0 and gt_b.sum() == 0: return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0: return 0.0

    skel_pred = skeletonize(pred_b)
    skel_gt   = skeletonize(gt_b)

    tprec_num = np.logical_and(skel_pred, gt_b).sum()
    tprec_den = skel_pred.sum()
    tprec = tprec_num / tprec_den if tprec_den > 0 else 0.0

    tsens_num = np.logical_and(skel_gt, pred_b).sum()
    tsens_den = skel_gt.sum()
    tsens = tsens_num / tsens_den if tsens_den > 0 else 0.0

    if tprec + tsens == 0: return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)

def generate_error_map(pred_bool, gt_bool):
    """
    生成直观的误差图：
    绿色: 预测正确 (TP)
    红色: 误检，导致 Precision 下降 (FP)
    蓝色: 漏检，导致 Recall 下降 (FN)
    """
    h, w = pred_bool.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    vis[pred_bool & gt_bool] = [0, 255, 0]   # 绿色 TP
    vis[pred_bool & ~gt_bool] = [255, 0, 0]  # 红色 FP
    vis[~pred_bool & gt_bool] = [0, 0, 255]  # 蓝色 FN
    
    return vis

# ==========================================
# 🚀 3. 主循环与统计
# ==========================================
def main():
    gt_paths = glob.glob(os.path.join(gt_dir, "*.png"))
    print(f"🔍 找到 {len(gt_paths)} 张 Ground Truth 图片，开始评估并挖掘典型图片...\n")

    metrics = {
        "base": {"iou": [], "pre": [], "rec": [], "cldice": []},
        "dir":  {"iou": [], "pre": [], "rec": [], "cldice": []}
    }

    missing_files = 0
    saved_visuals_count = 0

    for gt_path in tqdm(gt_paths, desc="Evaluating"):
        base_name = os.path.basename(gt_path)
        
        path_pred_base = os.path.join(pred_base_dir, base_name)
        path_pred_dir  = os.path.join(pred_dir_dir, base_name)

        if not os.path.exists(path_pred_base) or not os.path.exists(path_pred_dir):
            missing_files += 1
            continue

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_base = cv2.imread(path_pred_base, cv2.IMREAD_GRAYSCALE)
        pred_dir  = cv2.imread(path_pred_dir, cv2.IMREAD_GRAYSCALE)

        valid_mask = (gt_img != 255)
        gt_fg   = (gt_img > 0)
        base_fg = (pred_base > 0)
        dir_fg  = (pred_dir > 0)

        # 指标计算
        b_iou, b_pre, b_rec = compute_basic_metrics(base_fg, gt_fg, valid_mask)
        b_cld = compute_cldice(base_fg, gt_fg, valid_mask)
        metrics["base"]["iou"].append(b_iou); metrics["base"]["pre"].append(b_pre)
        metrics["base"]["rec"].append(b_rec); metrics["base"]["cldice"].append(b_cld)

        d_iou, d_pre, d_rec = compute_basic_metrics(dir_fg, gt_fg, valid_mask)
        d_cld = compute_cldice(dir_fg, gt_fg, valid_mask)
        metrics["dir"]["iou"].append(d_iou); metrics["dir"]["pre"].append(d_pre)
        metrics["dir"]["rec"].append(d_rec); metrics["dir"]["cldice"].append(d_cld)

        # ==========================================
        # 💡 核心逻辑：拦截“Recall大幅提升，但Precision明显下降”的图片
        # 设定阈值：Recall 提升 > 3%，并且 Precision 下降 > 2%
        # ==========================================
        diff_rec = d_rec - b_rec
        diff_pre = d_pre - b_pre # 注意这是负数
        
        if diff_rec > 0.03 and diff_pre < -0.02:
            tqdm.write(f"\n🎯 发现典型 Trade-off 图片: {base_name} | Precision: {diff_pre*100:.1f}% | Recall: +{diff_rec*100:.1f}%")
            
            # 限制保存数量，防止硬盘爆满，最多找 15 张够写论文了
            if saved_visuals_count < 150:
                # 绘制三图对比：原图GT，Baseline误差图，方向场误差图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"Trade-off Analysis: {base_name}\nGreen: Correct (TP) | Red: Over-segmentation (FP) | Blue: Missed/Broken (FN)", fontsize=14, fontweight='bold')

                # 1. Ground Truth
                axes[0].imshow(gt_img, cmap='gray')
                axes[0].set_title("Ground Truth Mask")
                axes[0].axis('off')

                # 2. Baseline
                err_base = generate_error_map(base_fg, gt_fg)
                axes[1].imshow(err_base)
                axes[1].set_title(f"Baseline\nPre: {b_pre*100:.1f}% | Rec: {b_rec*100:.1f}% | clDice: {b_cld*100:.1f}%")
                axes[1].axis('off')

                # 3. Ours
                err_dir = generate_error_map(dir_fg, gt_fg)
                axes[2].imshow(err_dir)
                axes[2].set_title(f"Ours (Directional Field)\nPre: {d_pre*100:.1f}% | Rec: {d_rec*100:.1f}% | clDice: {d_cld*100:.1f}%")
                axes[2].axis('off')

                plt.tight_layout()
                save_path = os.path.join(out_visual_dir, f"tradeoff_{base_name}")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                
                saved_visuals_count += 1

    # ... 保持原有的总体性能打印部分不变 ...
    print("\n" + "="*65)
    print("📈 评估完成！请去文件夹查看生成的 Trade-off 分析图：")
    print(out_visual_dir)
    print("="*65)

if __name__ == "__main__":
    main()