import os
import cv2
import numpy as np

# ------------------- 全局变量配置 -------------------
# 1. 想要挑选展示的图片名（不要后缀）
TARGET_NAMES = ['0122_001', '0037_011', '0008_004', '0167_001']

# 2. 文件夹路径配置
IMG_ROOT = r'VOCdevkit\VOC2007\JPEGImages'
GT_ROOT = r'VOCdevkit\VOC2007\SegmentationClass'

# 输入：模型预测图所在的根目录
MIOU_BASE = 'miou_out'
MODELS = {
    'BaseLine': 'miou_unet_100epoch',
    'AttUnet': 'miou_attunet',
    'DeepLabv3Plus': 'miou_deeplabv3plus',
    'UnetPlusPlus': 'miou_unetplusplus',
    'Ours': 'miou_unet_dir_cbam'
}

# 3. 输出目标文件夹
SAVE_ROOT = 'comparison_results2'

# ---------------------------------------------------

def process_and_save():
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    print("=====================================================")
    print(" 🖱️  交互选框提示：")
    print(" 1. 弹出图片窗口后，按住鼠标左键拖拽画框。")
    print(" 2. 画好后，按下键盘的【空格键 (Space)】或【回车键 (Enter)】确认。")
    print(" 3. 如果画错了想重新画，用鼠标重新拖拽即可。")
    print(" 4. 如果这张图不想画框，直接按【空格键】跳过。")
    print("=====================================================\n")

    for img_name in TARGET_NAMES:
        print(f"📌 正在处理图片: {img_name}")
        
        # --- A. 处理原图与交互式选框 ---
        src_img_path = os.path.join(IMG_ROOT, img_name + '.jpg')
        roi_rect = (0, 0, 0, 0) # 默认 ROI: x, y, w, h
        
        if os.path.exists(src_img_path):
            img = cv2.imread(src_img_path)
            
            # 弹出交互式窗口让用户选框
            window_name = f"Select ROI for {img_name} (Press SPACE to confirm)"
            # 参数说明：showCrosshair=True 显示十字准星, fromCenter=False 从左上角开始画
            roi_rect = cv2.selectROI(window_name, img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(window_name) # 选完后关闭窗口
            
            x, y, w, h = roi_rect
            if w > 0 and h > 0:
                print(f"   ✅ 已选择关注区域: 坐标({x}, {y}), 宽{w}, 高{h}")
            else:
                print("   ⚠️ 未选择关注区域，将不绘制红框。")

            # 原图直接保存（不带红框，如果你原图也想带红框，把画框代码移到这里即可）
            save_path = os.path.join(SAVE_ROOT, f"{img_name}_Image.png")
            cv2.imwrite(save_path, img)
        else:
            print(f"   [Error] 找不到原图: {src_img_path}")
            continue # 如果原图都没有，后面的对比也没意义了，直接跳过

        # --- B. 处理GT ---
        gt_path = os.path.join(GT_ROOT, img_name + '.png')
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt.max() < 10: 
                gt = (gt * (255 // (gt.max() if gt.max() > 0 else 1))).astype(np.uint8)
            cv2.imwrite(os.path.join(SAVE_ROOT, f"{img_name}_GT.png"), gt)

        # --- C. 处理各个模型的预测图 ---
        for model_nick, folder_name in MODELS.items():
            pred_path = os.path.join(MIOU_BASE, folder_name, "detection-results", img_name + '.png')
            
            if os.path.exists(pred_path):
                # 以灰度图模式读取预测结果
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                
                # 亮度拉伸映射
                if pred.max() < 10: 
                    pred = (pred * (255 // (pred.max() if pred.max() > 0 else 1))).astype(np.uint8)
                
                save_name = f"{img_name}_{model_nick}.png"
                
                # 💡 核心逻辑：只给 BaseLine 和 Ours 画红框
                if model_nick in ['BaseLine', 'Ours'] and w > 0 and h > 0:
                    # 灰度图无法显示红色，必须先转成 BGR 三通道
                    pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
                    # 绘制红色矩形框，(0, 0, 255) 是 BGR 的红色，线宽为 3
                    cv2.rectangle(pred_color, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
                    # 保存带红框的彩色图
                    cv2.imwrite(os.path.join(SAVE_ROOT, save_name), pred_color)
                else:
                    # 其他模型直接保存原本的灰度图
                    cv2.imwrite(os.path.join(SAVE_ROOT, save_name), pred)
            else:
                print(f"   [Warning] 找不到预测图: {pred_path}")

    print("-" * 30)
    print(f"🎉 处理完成！所有对比图已保存在: {SAVE_ROOT}")

if __name__ == '__main__':
    process_and_save()