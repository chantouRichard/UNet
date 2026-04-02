import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

def compute_metrics(gt_dir, pred_dir, num_classes=2, target_id=1):
    """
    计算 mIoU 和 mPA
    target_id: 只有该像素值会被视为前景，其余均为背景
    """
    all_confusion_matrix = np.zeros((num_classes, num_classes))
    
    # 获取预测文件夹下的所有图片名
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
    
    for file_name in pred_files:
        # 读取预测图和GT
        pred_path = os.path.join(pred_dir, file_name)
        gt_path = os.path.join(gt_dir, file_name)
        
        if not os.path.exists(gt_path):
            continue
            
        # 转换为单通道灰度数组
        pred = np.array(Image.open(pred_path).convert('L'))
        gt = np.array(Image.open(gt_path).convert('L'))

        # --- 核心处理逻辑 ---
        # 预测图：非0即1处理（假设预测图已经是二值或0/255）
        pred = (pred > 0).astype(int)
        
        # 虚拟/现实GT：仅保留像素为 target_id (1) 的部分，其余转为 0
        gt_cleaned = np.where(gt == target_id, 1, 0)

        # 计算单张图的混淆矩阵并累加
        # flatten后计算，类别为[0, 1]
        mask = (gt_cleaned >= 0) & (gt_cleaned < num_classes)
        label = num_classes * gt_cleaned[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=num_classes**2)
        all_confusion_matrix += count.reshape(num_classes, num_classes)

    # 从混淆矩阵计算指标
    # PA (Pixel Accuracy) 每类的比例
    tp = np.diag(all_confusion_matrix)
    pos_gt = all_confusion_matrix.sum(axis=1)
    pos_pred = all_confusion_matrix.sum(axis=0)
    
    # 计算 PA (每一类的精度)
    class_pa = tp / np.maximum(pos_gt, 1e-6)
    mpa = np.mean(class_pa)
    
    # 计算 IoU: TP / (TP + FP + FN)
    union = pos_gt + pos_pred - tp
    class_iou = tp / np.maximum(union, 1e-6)
    miou = np.mean(class_iou)

    return miou, mpa, class_iou[1], class_pa[1]

# ------------------- 路径配置 -------------------
configs = {
    "Virtual Data (虚拟数据)": {
        "pred": r"miou_out\miou_CBAM_best\detection-results",
        "gt": r"VOCdevkit\VOC2007-virtual\SegmentationClass"
    },
    "Real Data (现实数据)": {
        "pred": r"miou_out\miou_unet_cbam\detection-results",
        "gt": r"VOCdevkit\VOC2007\SegmentationClass"
    }
}

# ------------------- 执行计算 -------------------
print("="*50)
print(f"{'Experiment Group':<25} | {'mIoU':<8} | {'mPA':<8}")
print("-"*50)

for name, paths in configs.items():
    try:
        miou, mpa, fg_iou, fg_pa = compute_metrics(paths["gt"], paths["pred"])
        print(f"{name:<25} | {miou:>7.2%} | {mpa:>7.2%}")
        # 如果需要单独看绳索（前景）的指标，可以打印下面这行
        print(f"  -> Foreground (Cable): IoU={fg_iou:.2%}, PA={fg_pa:.2%}")
    except Exception as e:
        print(f"{name:<25} | 计算出错: {e}")

print("="*50)