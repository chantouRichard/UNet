import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        # 1. 读取原图、标签图 和 我们预生成的 Vessel 先验图
        jpg   = Image.open(os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg"))
        png   = Image.open(os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png"))
        # 假设你把生成的先验图放在了 VesselMasks 文件夹
        prior = Image.open(os.path.join(self.dataset_path, "VOC2007/VesselMasks", name + ".png"))

        # 2. 数据增强 (重点：将 prior 加入增强，确保它和 jpg/png 的旋转平移同步)
        # 你需要微调你的 get_random_data 函数，让它支持同时传入三个 Image 对象
        jpg, png, prior = self.get_random_data(jpg, png, prior, self.input_shape, random = self.train)

        # 3. 预处理原图
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        
        # 4. 预处理标签
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        # 5. 预处理 Vessel 先验 (归一化到 0-1 供注意力机制使用)
        # 我们通常希望注意力权重是 float32 类型
        prior_np = np.array(prior, np.float32) / 255.0
        prior_np = np.expand_dims(prior_np, 0) # 变成 [1, H, W] 通道格式

        # 转化 one_hot
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # 6. 返回结果，增加 prior
        return jpg, png, seg_labels, prior_np

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, prior, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        # prior 已经是单通道灰度图，确保它是 L 模式
        prior   = prior.convert("L") 
        
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            # 三者同步缩放
            image   = image.resize((nw,nh), Image.BICUBIC)
            label   = label.resize((nw,nh), Image.NEAREST)
            prior   = prior.resize((nw,nh), Image.NEAREST)

            # 三者同步填充灰边/黑边
            new_image = Image.new('RGB', [w, h], (128,128,128))
            new_label = Image.new('L', [w, h], (0))
            new_prior = Image.new('L', [w, h], (0))

            dx, dy = (w-nw)//2, (h-nh)//2
            new_image.paste(image, (dx, dy))
            new_label.paste(label, (dx, dy))
            new_prior.paste(prior, (dx, dy))
            
            return new_image, new_label, new_prior

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
            
        # 三者同步 Resize
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        prior = prior.resize((nw,nh), Image.NEAREST)
        
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            prior = prior.transpose(Image.FLIP_LEFT_RIGHT)
        
        # ------------------------------------------#
        #   随机平移与填充
        # ------------------------------------------#
        # 修正 dx, dy 的随机范围，防止超出边界
        dx = int(self.rand(0, max(0, w-nw)))
        dy = int(self.rand(0, max(0, h-nh)))
        
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_prior = Image.new('L', (w,h), (0))
        
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        new_prior.paste(prior, (dx, dy))
        
        image = new_image
        label = new_label
        prior = new_prior

        # ---------------------------------#
        #   对图像进行色域变换 (仅限原图)
        # ---------------------------------#
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        
        hue_ch, sat_ch, val_ch = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue_ch, lut_hue), cv2.LUT(sat_ch, lut_sat), cv2.LUT(val_ch, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label, prior

# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    vessels     = []
    for img, png, labels, vsl in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        vessels.append(vsl)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    vessels  = torch.from_numpy(np.array(vessels)).type(torch.FloatTensor)
    return images, pngs, seg_labels, vessels
