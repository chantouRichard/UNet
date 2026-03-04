import torch
import torch.nn as nn

# ===== 1. 导入你的模型 =====
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nets.mkunet_network import MK_UNet   # 改成你真实的文件名

# ===== 2. 设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 3. 构造模型 =====
model = MK_UNet(num_classes=1, in_channels=3)
model = model.to(device)
model.train()

# ===== 4. 构造假的输入数据 =====
# 假设输入尺寸 256x256
batch_size = 2
images = torch.randn(batch_size, 3, 256, 256).to(device)

# segmentation label
labels = torch.randn(batch_size, 1, 256, 256).to(device)

# ===== 5. 定义 loss =====
criterion = nn.MSELoss()   # 这里只是测试是否能跑

# ===== 6. 前向传播 =====
outputs = model(images)

# 如果模型返回 list
if isinstance(outputs, list):
    outputs = outputs[0]

print("Output shape:", outputs.shape)

# ===== 7. 计算 loss =====
loss = criterion(outputs, labels)

print("Loss:", loss.item())

# ===== 8. 反向传播测试 =====
loss.backward()

print("Backward success ✅")