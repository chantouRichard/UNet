import torch
import sys
import numpy as np

def load_model(pth_path):
    print(f"Loading: {pth_path}")
    checkpoint = torch.load(pth_path, map_location="cpu")

    # 兼容不同保存方式
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    return state_dict

def remove_module_prefix(state_dict):
    """去掉 DataParallel 的 module. 前缀"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def check_alpha(state_dict):
    print("\nSearching for alpha parameters...\n")
    found = False

    for name, param in state_dict.items():
        if "alpha" in name.lower():
            found = True
            param_np = param.cpu().numpy()

            print("="*60)
            print(f"Parameter name : {name}")
            print(f"Shape          : {param_np.shape}")
            print(f"Mean           : {param_np.mean():.6f}")
            print(f"Min            : {param_np.min():.6f}")
            print(f"Max            : {param_np.max():.6f}")
            print(f"Values         :\n{param_np}")
            print("="*60)

    if not found:
        print("❌ No alpha parameter found in this model.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_alpha.py your_model.pth")
        sys.exit(1)

    pth_path = sys.argv[1]
    state_dict = load_model(pth_path)
    state_dict = remove_module_prefix(state_dict)
    check_alpha(state_dict)