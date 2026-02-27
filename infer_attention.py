import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from nets.unet import Unet  # 根据你的工程路径修改


# ===============================
# Hook 用于抓取 CBAM 注意力
# ===============================
attention_maps = []

def hook_fn(module, input, output):
    # output shape: [B,C,H,W]
    att = output.detach().cpu()
    attention_maps.append(att)


def load_model(model_path, device):
    model = Unet(num_classes=2)  # 根据你的类别数修改
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(image).unsqueeze(0)
    return image, img


def visualize_attention(original_img, attention_tensor, save_path):
    att = attention_tensor[0]  # [C,H,W]
    att = torch.mean(att, dim=0)  # 通道平均
    att = att.numpy()

    att = (att - att.min()) / (att.max() - att.min() + 1e-6)
    att = cv2.resize(att, original_img.size)
    att = np.uint8(255 * att)

    heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)

    original_np = np.array(original_img)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)
    print(f"结果已保存到: {save_path}")


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    # 注册 hook（抓第一个CBAM）
    model.vgg.cbam1.register_forward_hook(hook_fn)

    original_img, img_tensor = preprocess_image(args.image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)

    if len(attention_maps) == 0:
        print("没有捕获到注意力图，请检查CBAM名称是否正确")
        return

    visualize_attention(original_img, attention_maps[0], args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="attention_result.png")

    args = parser.parse_args()
    main(args)