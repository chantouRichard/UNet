import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

class SegmentationAugmentor:
    """
    图像 + Mask 同步增强器：固定生成原图、水平翻转、光照调节三个版本
    """
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.img_out = os.path.join(out_dir, "img")
        self.mask_out = os.path.join(out_dir, "masks")

        os.makedirs(self.img_out, exist_ok=True)
        os.makedirs(self.mask_out, exist_ok=True)

        # 定义具体的变换，不再随机
        self.flip_tfm = A.HorizontalFlip(p=1.0)
        self.light_tfm = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
        ])

    def save_pair(self, img, mask, base_name, suffix):
        """保存图像对的辅助函数"""
        save_name = f"{base_name}_{suffix}.png"
        cv2.imwrite(
            os.path.join(self.img_out, save_name),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(self.mask_out, save_name),
            mask.astype(np.uint8)
        )

    def augment_one(self, img_path, mask_path):
        """
        固定生成 3 个版本：
        1. 原图 (original)
        2. 水平翻转 (flip)
        3. 光照调节 (light)
        """
        # 读取
        img = cv2.imread(img_path)
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: return

        base = os.path.splitext(os.path.basename(img_path))[0]

        # 1. 保存原图版本
        self.save_pair(img_rgb, mask, base, "orig")

        # 2. 保存水平翻转版本
        flipped = self.flip_tfm(image=img_rgb, mask=mask)
        self.save_pair(flipped["image"], flipped["mask"], base, "flip")

        # 3. 保存光照调节版本
        lighted = self.light_tfm(image=img_rgb, mask=mask)
        self.save_pair(lighted["image"], lighted["mask"], base, "light")

    def run(self, img_dir, mask_dir):
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for fname in tqdm(img_files, desc="Processing"):
            img_path = os.path.join(img_dir, fname)
            base = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, base + ".png")

            if not os.path.exists(mask_path):
                continue

            self.augment_one(img_path, mask_path)

        print(f"✨ 处理完成！每个原始样本已生成 3 个固定版本（原图/翻转/光照）。")
        print(f"总输出文件数: {len(img_files) * 3}")

if __name__ == "__main__":
    IMAGE_DIR = r"img\\img"
    MASK_DIR  = r"img\\masks"
    OUT_DIR   = r"img\\img_aug"

    augmentor = SegmentationAugmentor(OUT_DIR)
    # 注意：这里不再需要传入 AUG_NUM，因为逻辑是固定的 3 张
    augmentor.run(IMAGE_DIR, MASK_DIR)