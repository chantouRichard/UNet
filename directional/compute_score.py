import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize

# ==========================================
# 📂 1. 路径配置
# ==========================================
project_root = r"E:\03_Learning\MachineLearning\unet-pytorch"

gt_dir       = os.path.join(project_root, "VOCdevkit", "VOC2007", "SegmentationClass")
pred_base_dir= os.path.join(project_root, "miou_out", "miou_unet_100epoch", "detection-results") # Baseline
pred_dir_dir = os.path.join(project_root, "miou_out", "miou_unet_dir_cbam", "detection-results")  # 方向场创新模型

# ==========================================
# 🧮 2. 评估指标计算函数
# ==========================================
def compute_basic_metrics(pred, gt, valid_mask):
    """同时计算 IoU, Precision, Recall (只关注前景类别)"""
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
    """计算 clDice (Centerline Dice) - 评估拓扑连通度"""
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

# ==========================================
# 🚀 3. 主循环与统计
# ==========================================
def main():
    gt_paths = glob.glob(os.path.join(gt_dir, "*.png"))
    print(f"🔍 找到 {len(gt_paths)} 张 Ground Truth 图片，开始全面对比评估...\n")

    # 统计累加器
    metrics = {
        "base": {"iou": [], "pre": [], "rec": [], "cldice": []},
        "dir":  {"iou": [], "pre": [], "rec": [], "cldice": []}
    }

    missing_files = 0

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

        # Baseline 指标
        b_iou, b_pre, b_rec = compute_basic_metrics(base_fg, gt_fg, valid_mask)
        b_cld = compute_cldice(base_fg, gt_fg, valid_mask)
        
        metrics["base"]["iou"].append(b_iou)
        metrics["base"]["pre"].append(b_pre)
        metrics["base"]["rec"].append(b_rec)
        metrics["base"]["cldice"].append(b_cld)

        # Directional Field 指标
        d_iou, d_pre, d_rec = compute_basic_metrics(dir_fg, gt_fg, valid_mask)
        d_cld = compute_cldice(dir_fg, gt_fg, valid_mask)
        
        metrics["dir"]["iou"].append(d_iou)
        metrics["dir"]["pre"].append(d_pre)
        metrics["dir"]["rec"].append(d_rec)
        metrics["dir"]["cldice"].append(d_cld)

        # 💡 高能预警：打印提升极其明显的图片！
        # 设定阈值：clDice 提升 > 2% 或 Recall 提升 > 2%
        if (d_cld - b_cld) > 0.02 or (d_rec - b_rec) > 0.02:
            tqdm.write(f"🌟 显著提升发现: {base_name} | clDice: +{(d_cld - b_cld)*100:.1f}% | Recall: +{(d_rec - b_rec)*100:.1f}%")

    # ==========================================
    # 📊 4. 打印极其帅气的对比报告
    # ==========================================
    if missing_files > 0:
        print(f"\n⚠️ 警告: 有 {missing_files} 个文件在预测文件夹中找不到对应的结果，已跳过。")

    print("\n" + "="*65)
    print(f"{'🏆 最终多维评估对比报告 (Baseline vs Directional Field) 🏆':^61}")
    print("="*65)
    
    def avg(lst): return np.mean(lst) * 100 if len(lst) > 0 else 0

    # 计算均值
    m_base = {k: avg(v) for k, v in metrics["base"].items()}
    m_dir  = {k: avg(v) for k, v in metrics["dir"].items()}

    print(f"{'Metric (Foreground)':<22} | {'Baseline (unet)':<18} | {'Proposed (direction)':<20}")
    print("-" * 65)
    
    metrics_list = [
        ("Foreground IoU", "iou"),
        ("Recall (Sensitivity)", "rec"),
        ("Precision", "pre"),
        ("clDice (Topology)", "cldice")
    ]

    for display_name, key in metrics_list:
        val_base = m_base[key]
        val_dir  = m_dir[key]
        diff = val_dir - val_base
        sign = "+" if diff > 0 else ""
        print(f"{display_name:<22} | {val_base:>7.2f}% {' ':<10} | {val_dir:>7.2f}% ({sign}{diff:.2f}%)")

    print("="*65)
    print("💡 提示：快去查看上面带 🌟 标志的图片，这些是证明方向场保拓扑、防断裂的最佳素材图！")

if __name__ == "__main__":
    main()