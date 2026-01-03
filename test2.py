# debug_train.py
import torch
import time
from nets.unet import Unet
from utils.dataloader import UnetDataset, unet_dataset_collate

def test_data_loading():
    """测试数据加载"""
    print("=" * 50)
    print("测试数据加载...")
    print("=" * 50)
    
    try:
        # 创建一个最小的数据集
        from torch.utils.data import DataLoader
        
        dataset = UnetDataset(...)  # 用你的参数
        
        # 测试单个样本
        start = time.time()
        img, mask = dataset[0]
        print(f"加载单个样本耗时: {time.time()-start:.2f}秒")
        print(f"图像形状: {img.shape}, 掩码形状: {mask.shape}")
        
        # 测试DataLoader
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
        
        start = time.time()
        for i, batch in enumerate(dataloader):
            if i >= 2:  # 只测试2个batch
                break
            print(f"Batch {i}: {batch[0].shape}, 加载耗时: {time.time()-start:.2f}秒")
            start = time.time()
            
    except Exception as e:
        print(f"数据加载错误: {e}")
        import traceback
        traceback.print_exc()

def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 50)
    print("测试模型前向传播...")
    print("=" * 50)
    
    try:
        model = Unet(num_classes=2, backbone="vgg")
        model.eval()
        
        # 测试小尺寸输入
        print("测试512×512输入...")
        x = torch.randn(1, 3, 512, 512)
        
        start = time.time()
        with torch.no_grad():
            output = model(x)
        print(f"前向传播耗时: {time.time()-start:.2f}秒")
        print(f"输出形状: {output.shape}")
        
        # 测试大尺寸输入（你的实际尺寸）
        print("\n测试1024×1024输入...")
        x = torch.randn(1, 3, 1024, 1024)
        
        start = time.time()
        with torch.no_grad():
            output = model(x)
        print(f"前向传播耗时: {time.time()-start:.2f}秒")
        
        # 检查显存
        if torch.cuda.is_available():
            print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
    except Exception as e:
        print(f"模型前向传播错误: {e}")
        import traceback
        traceback.print_exc()

def test_training_step():
    """测试单步训练"""
    print("\n" + "=" * 50)
    print("测试单步训练...")
    print("=" * 50)
    
    try:
        model = Unet(num_classes=2, backbone="vgg")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 小批量数据
        x = torch.randn(2, 3, 256, 256)
        y = torch.randint(0, 2, (2, 256, 256)).long()
        
        print("开始训练步骤...")
        start = time.time()
        
        # 前向
        outputs = model(x)
        
        # 计算损失
        loss = criterion(outputs, y)
        print(f"损失计算完成: {loss.item():.4f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        print(f"单步训练总耗时: {time.time()-start:.2f}秒")
        
    except Exception as e:
        print(f"训练步骤错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始诊断训练卡住问题...")
    
    # 按顺序测试
    test_data_loading()
    test_model_forward()
    test_training_step()
    
    print("\n诊断完成！")