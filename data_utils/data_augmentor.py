import os
import cv2
import random
import numpy as np
import albumentations as A
from tqdm import tqdm

class SegmentationAugmentor:
    """
    图像 + Mask 同步增强器（语义分割）
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.img_out = os.path.join(out_dir, "JPEGImages")
        self.mask_out = os.path.join(out_dir, "SegmentationClass")

        os.makedirs(self.img_out, exist_ok=True)
        os.makedirs(self.mask_out, exist_ok=True)

        self.transforms = self._build_transforms()

    def _build_transforms(self):
        """
        多套增强策略，随机选用
        """
        return [
            # 1. 轻量几何
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.02,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.4
                ),
            ]),

            # 2. 光照变化
            A.Compose([
                A.RandomBrightnessContrast(0.2, 0.2, p=0.8),
                A.CLAHE(clip_limit=2.0, p=0.3),
            ]),

            # 3. 噪声 & 模糊
            A.Compose([
                A.GaussNoise(var_limit=(10, 40), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
            ]),

            # 4. 综合增强（中等强度）
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
                A.Rotate(limit=5, p=0.4),
                A.GaussNoise(var_limit=(5, 25), p=0.3),
            ]),
        ]

    def augment_one(self, img_path, mask_path, n=4):
        """
        对单张图像生成 n 个增强版本
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        base = os.path.splitext(os.path.basename(img_path))[0]

        for i in range(n):
            tfm = random.choice(self.transforms)

            augmented = tfm(image=img, mask=mask)
            aug_img = augmented["image"]
            aug_mask = augmented["mask"]

            # 确保 mask 为 uint8（不插值）
            aug_mask = aug_mask.astype(np.uint8)

            img_name = f"{base}_aug_{i+1:03d}.png"
            mask_name = f"{base}_aug_{i+1:03d}.png"

            cv2.imwrite(
                os.path.join(self.img_out, img_name),
                cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(self.mask_out, mask_name),
                aug_mask
            )

    def run(self, img_dir, mask_dir, n_per_image=4):
        """
        批量增强
        """
        img_files = sorted(os.listdir(img_dir))

        for fname in tqdm(img_files, desc="Augmenting"):
            img_path = os.path.join(img_dir, fname)
            base = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, base + ".png")

            if not os.path.exists(mask_path):
                print(f"[跳过] 未找到 mask: {fname}")
                continue

            self.augment_one(img_path, mask_path, n=n_per_image)

        print("增强完成！")
        print("Images:", self.img_out)
        print("Masks :", self.mask_out)


if __name__ == "__main__":
    # ========= 修改为你的路径 =========
    IMAGE_DIR = r"VOCdevkit\\VOC2007-temp\\JPEGImages"
    MASK_DIR  = r"VOCdevkit\\VOC2007-temp\\SegmentationClass"
    OUT_DIR   = r"VOCdevkit\\VOC2007"
    AUG_NUM   = 4  # 每张图生成多少增强样本
    # =================================

    augmentor = SegmentationAugmentor(OUT_DIR)
    augmentor.run(IMAGE_DIR, MASK_DIR, AUG_NUM)
