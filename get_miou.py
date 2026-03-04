import os

from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results
from data_utils.visual_mask import batch_visualize

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 2
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    name_classes    = ["_background_","rope"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    import time
    from datetime import datetime

    # 获取当前时间，格式为：年_月_日_时_分_秒
    # 例如：2023_10_27_14_30_05
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # 使用 f-string 进行拼接
    miou_out_path = f"miou_out/miou_{time_str}"

    # 或者如果你希望放在 logs 文件夹下
    # miou_out_path = os.path.join("logs", f"miou_out_{time_str}")

    print(f"本次运行的 mIoU 输出路径为: {miou_out_path}")
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")
        
        # --- 新增：调用混合可视化函数 ---
        print("Generate overlap visualization...")
        # 定义叠加图的保存路径
        overlay_save_dir = os.path.join(miou_out_path, 'overlay_results')
        
        # 直接调用你的批量处理函数
        # 注意：IMG_DIR 指向你的原图路径，MASK_DIR 指向刚才生成的预测图路径
        batch_visualize(
            img_folder   = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages"), 
            mask_folder  = pred_dir, 
            save_dir     = overlay_save_dir, 
            show         = False
        )
        print(f"Overlap visualization saved to: {overlay_save_dir}")
        # ----------------------------------

    if miou_mode == 0 or miou_mode == 2:
        print("Get metrics.")
        # 1. 执行计算，获取混淆矩阵及基础指标
        # 注意：确保你的 compute_mIoU 返回了 hist (混淆矩阵)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        
        # 2. 从混淆矩阵中计算更多指标 (Sen, Spe, F1, Acc)
        # 假设 hist 是 [num_classes, num_classes] 的混淆矩阵
        # 针对分类任务，通常我们关注“绳索”(类别1) 的表现
        
        # 计算全局准确率 (Acc)
        import numpy as np
        Acc = np.diag(hist).sum() / hist.sum()
        
        # 针对每一类计算 Sen (Recall) 和 Spe
        # Sen (Sensitivity) 其实就是 PA_Recall
        Sen = PA_Recall 
        
        # 计算 Specificity (Spe)
        # 对于二分类，背景的 Spe 就是 绳索的 Recall，反之亦然
        # 这里展示通用的 Spe 计算逻辑
        Spe = []
        for i in range(num_classes):
            tp = hist[i, i]
            fp = hist[:, i].sum() - tp
            fn = hist[i, :].sum() - tp
            tn = hist.sum() - (tp + fp + fn)
            Spe.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
        
        # 计算 F1-Score
        # F1 = 2 * (Prec * Rec) / (Prec + Rec)
        F1 = []
        for i in range(num_classes):
            if (Precision[i] + PA_Recall[i]) != 0:
                f1 = 2 * Precision[i] * PA_Recall[i] / (Precision[i] + PA_Recall[i])
            else:
                f1 = 0
            F1.append(f1)

        # 3. 打印并记录结果
        print("Get metrics done.")
        
        # 保存到本地文件
        with open(os.path.join(miou_out_path, "detailed_metrics.txt"), 'w') as f:
            f.write(f"Evaluation Results (Time: {time_str})\n")
            f.write("-" * 50 + "\n")
            f.write(f"Overall Accuracy (Acc): {Acc * 100:.2f}%\n")
            for i, class_name in enumerate(name_classes):
                f.write(f"Class: {class_name}\n")
                f.write(f"  IoU: {IoUs[i] * 100:.2f}%\n")
                f.write(f"  Sen (Recall): {Sen[i] * 100:.2f}%\n")
                f.write(f"  Spe: {Spe[i] * 100:.2f}%\n")
                f.write(f"  F1-Score: {F1[i] * 100:.2f}%\n")
                f.write("-" * 20 + "\n")

        # 调用原有的展示函数
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        print(f"Detailed metrics saved to: {os.path.join(miou_out_path, 'detailed_metrics.txt')}")