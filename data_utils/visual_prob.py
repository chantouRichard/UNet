import os
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unet import Unet

def visualize_uncertainty(image_path, unet):
    # 1. 加载图片
    orig_image = Image.open(image_path)
    # 转换为 OpenCV 格式用于绘图
    vis_img = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
    
    # 2. 获取概率图 (0.0 - 1.0)
    prob_map = unet.get_prob_map(orig_image)
    
    # 3. 定义区间颜色 (BGR)
    # 红色: 强预测区 (Prob > 0.7)
    # 黄色: 犹豫/潜在误报区 (0.3 < Prob <= 0.7)
    # 蓝色: 弱信号区 (0.1 < Prob <= 0.3)
    
    overlay = vis_img.copy()
    
    # 涂色逻辑
    overlay[prob_map > 0.7] = [0, 0, 255]    # 红色
    overlay[(prob_map > 0.3) & (prob_map <= 0.7)] = [0, 255, 255] # 黄色
    overlay[(prob_map > 0.1) & (prob_map <= 0.3)] = [255, 0, 0]   # 蓝色
    overlay[(prob_map <= 0.1)] = [255, 255, 0]   # 蓝色

    # 4. 叠加到原图
    alpha = 0.4  # 透明度
    res_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)
    
    # 5. 添加图例文字
    cv2.putText(res_img, "Red: Prob > 0.7 (Strong)", (10, 30), 1, 1.2, (0,0,255), 2)
    cv2.putText(res_img, "Yellow: 0.3-0.7 (Uncertain)", (10, 60), 1, 1.2, (0,255,255), 2)
    cv2.putText(res_img, "Blue: 0.1-0.3 (Weak)", (10, 90), 1, 1.2, (255,0,0), 2)

    return res_img

if __name__ == "__main__":
    # 初始化网络
    unet = Unet()
    
    while True:
        img_p = input('请输入图片路径 (输入 q 退出): ')
        if img_p.lower() == 'q': break
        
        if not os.path.exists(img_p):
            print("路径不存在！")
            continue
            
        # 1. 获取可视化结果 (BGR 格式)
        result_bgr = visualize_uncertainty(img_p, unet)
        
        # 2. 转换：BGR -> RGB (因为 OpenCV 是 BGR，而 PIL/系统查看器是 RGB)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. 转为 PIL 对象并直接弹窗显示
        final_pil = Image.fromarray(result_rgb)
        final_pil.show()  # <--- 这一行替代了 cv2.imshow，简单直接