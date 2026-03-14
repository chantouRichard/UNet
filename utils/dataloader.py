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

        #-------------------------------#
        #   从文件中读取图像、标签 和 Vessel先验图
        #-------------------------------#
        jpg    = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png    = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        # 读取先验图并强制转换为灰度图 ('L' 模式)
        vessel = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/Vessel_Only"), name + ".png")).convert('L')
        
        #-------------------------------#
        #   数据增强 (注意这里多传了一个 vessel)
        #-------------------------------#
        jpg, png, vessel = self.get_random_data(jpg, png, vessel, self.input_shape, random = self.train)

        # 处理原图 (RGB)
        jpg = preprocess_input(np.array(jpg, np.float64)) # shape: (H, W, 3)
        
        # 处理 Vessel图：将 0/255 的值归一化为 0.0/1.0，并增加通道维度
        vessel_np = np.array(vessel, np.float64) / 255.0 
        vessel_np = np.expand_dims(vessel_np, axis=-1)    # shape: (H, W, 1)
        
        # 在通道维度拼接 (H, W, 3) + (H, W, 1) -> (H, W, 4)
        jpg_4c = np.concatenate([jpg, vessel_np], axis=-1)
        
        # 转换为 PyTorch 需要的 (C, H, W) -> (4, H, W)
        jpg_4c = np.transpose(jpg_4c, [2, 0, 1])

        # 处理标签
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # 注意：这里返回的 jpg_4c 已经是 4通道了，变量名没改是为了兼容你后面的 collate_fn
        return jpg_4c, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, vessel, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        # vessel 已经是 'L' 模式，不需要额外转换

        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            
            # 同步处理 vessel
            vessel      = vessel.resize((nw,nh), Image.NEAREST)
            new_vessel  = Image.new('L', [w, h], (0))
            new_vessel.paste(vessel, ((w-nw)//2, (h-nh)//2))
            
            return new_image, new_label, new_vessel

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
            
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        vessel = vessel.resize((nw,nh), Image.NEAREST) # 同步缩放
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image  = image.transpose(Image.FLIP_LEFT_RIGHT)
            label  = label.transpose(Image.FLIP_LEFT_RIGHT)
            vessel = vessel.transpose(Image.FLIP_LEFT_RIGHT) # 同步翻转
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条 (Padding)
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        
        new_image  = Image.new('RGB', (w,h), (128,128,128))
        new_label  = Image.new('L', (w,h), (0))
        new_vessel = Image.new('L', (w,h), (0))
        
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        new_vessel.paste(vessel, (dx, dy)) # 同步 Padding
        
        image  = new_image
        label  = new_label
        vessel = new_vessel

        #---------------------------------#
        #   色域变换 (注意：仅对 RGB 图像 image_data 生效，不影响 label 和 vessel)
        #---------------------------------#
        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_v, sat_v, val_v = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue_v, lut_hue), cv2.LUT(sat_v, lut_sat), cv2.LUT(val_v, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        # 返回增强后的三个图
        return image_data, label, vessel

# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
