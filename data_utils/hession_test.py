import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. 路径配置 (请确保路径正确) ---
# 预测图目录 (包含 .png 或 .jpg)
pred_dir = r'miou_out\miou_2026_02_11_20_52_28-CBAM+50\detection-results'
# GT标签目录 (VOC格式，通常目标为1，背景为0)
gt_dir = r'VOCdevkit\VOC2007\SegmentationClass'
# Hessian/Frangi处理结果目录 (JPG格式，0-255)
vessel_dir = r'Vessel_result'
# 结果保存目录
save_dir = r'Hessian_Analysis_Visual'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def run_analysis():
    # 获取预测结果文件列表
    file_list = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    fn_intensities = []  # 记录Hessian在漏检区的强度
    bg_intensities = []  # 记录Hessian在背景区的强度

    print(f"正在处理 {len(file_list)} 张图片...")

    for file_name in tqdm(file_list):
        # --- 2. 文件名匹配逻辑 ---
        base_name = os.path.splitext(file_name)[0]
        
        pred_path = os.path.join(pred_dir, file_name)
        gt_path = os.path.join(gt_dir, base_name + ".png") # GT通常是png
        vessel_path = os.path.join(vessel_dir, base_name + ".jpg") # 你的Hessian是jpg

        if not (os.path.exists(gt_path) and os.path.exists(vessel_path)):
            continue

        # --- 3. 读取并统一尺度 ---
        # 读取为单通道灰度图
        img_pred = cv2.imread(pred_path, 0)
        img_gt = cv2.imread(gt_path, 0)
        img_vessel = cv2.imread(vessel_path, 0)

        # 处理 GT: 假设 1 是目标，255 是 VOC 的边缘忽略区
        # 我们只想要目标区域 (1)
        gt_bin = np.where(img_gt == 1, 255, 0).astype(np.uint8)
        
        # 处理 Pred: 如果是 0/1 则转为 0/255
        if img_pred.max() <= 1:
            pred_bin = (img_pred * 255).astype(np.uint8)
        else:
            _, pred_bin = cv2.threshold(img_pred, 127, 255, cv2.THRESH_BINARY)

        # --- 4. 提取 FN (漏检区域) ---
        # FN = GT中有(255)，但Pred中没有(0)
        fn_mask = cv2.subtract(gt_bin, pred_bin)
        fn_mask[fn_mask < 0] = 0 # 确保只保留漏检部分

        # --- 5. 计算定量指标 ---
        if np.any(fn_mask > 0):
            # Hessian 在漏检区域的平均亮度
            val_fn = np.mean(img_vessel[fn_mask > 0])
            # Hessian 在背景区域的平均亮度 (GT为0的地方)
            val_bg = np.mean(img_vessel[gt_bin == 0])
            fn_intensities.append(val_fn)
            bg_intensities.append(val_bg)

        # --- 6. 图像合成可视化 (BGR) ---
        # 创建一个彩色底图
        height, width = img_pred.shape
        vis_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        # R通道 (红色): 放置漏检区 (FN)
        vis_rgb[:, :, 2] = fn_mask 
        
        # G通道 (绿色): 放置 Hessian 响应
        # 这样：漏检且Hessian有反应的地方会变成 黄色 (R+G)
        vis_rgb[:, :, 1] = img_vessel
        
        # B通道 (蓝色): 放置原预测结果 (调暗一点作为背景)
        vis_rgb[:, :, 0] = pred_bin // 2

        # 保存结果
        cv2.imwrite(os.path.join(save_dir, f"check_{base_name}.jpg"), vis_rgb)

    # --- 7. 输出分析报告 ---
    if fn_intensities:
        m_fn = np.mean(fn_intensities)
        m_bg = np.mean(bg_intensities)
        lift = m_fn / (m_bg + 1e-6)
        
        print("\n" + "="*30)
        print(f"分析完成！结果已保存至: {save_dir}")
        print(f"Hessian在漏检区平均强度: {m_fn:.2f}")
        print(f"Hessian在背景区平均强度: {m_bg:.2f}")
        print(f"信号提升倍数 (Lift Ratio): {lift:.2f}x")
        print("="*30)
        
        if lift > 2.5:
            print("结论：黄色区域明显，Hessian信息非常丰富，强烈建议做后处理！")
        else:
            print("结论：提升倍数较低，可能Hessian噪声较多，需谨慎设计后处理逻辑。")
    else:
        print("未发现漏检区域或未匹配到文件。")

if __name__ == "__main__":
    run_analysis()