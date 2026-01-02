import numpy as np
from PIL import Image
import sys


def check_mask_values(mask_path):
    """
    检查掩码图片的像素值

    参数:
        mask_path: 掩码图片路径

    返回:
        像素值统计信息
    """
    try:
        # 1. 打开图片
        mask = Image.open(mask_path)
        print(f"图片信息: {mask_path}")
        print(f"  尺寸: {mask.size} (宽×高)")
        print(f"  模式: {mask.mode}")

        # 2. 转换为numpy数组
        mask_array = np.array(mask)

        # 3. 获取所有像素值
        unique_values = np.unique(mask_array)

        # 4. 显示统计信息
        print(f"  像素值数量: {len(unique_values)}种")
        print(f"  所有像素值: {unique_values}")

        # 5. 显示每个值的像素数量
        print(f"\n  详细统计:")
        for value in unique_values:
            count = np.sum(mask_array == value)
            percentage = (count / mask_array.size) * 100
            print(f"    值 {value}: {count}像素 ({percentage:.2f}%)")

        # 6. 判断是否为有效的掩码
        print(f"\n  判断结果:")
        if len(unique_values) == 2:
            # 二值掩码
            if 0 in unique_values:
                print(f"  ✅ 是有效的二值掩码 (0和{unique_values[1]})")
            else:
                print(f"  ⚠️ 是二值掩码，但不含0值")
        elif len(unique_values) > 2:
            # 多类别掩码
            if 0 in unique_values:
                print(f"  ✅ 是有效的多类别掩码 (0, {', '.join(map(str, unique_values[unique_values != 0]))})")
            else:
                print(f"  ⚠️ 是多类别掩码，但背景不是0")
        else:
            print(f"  ❓ 只有一个像素值: {unique_values[0]}")

        return unique_values

    except Exception as e:
        print(f"错误: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        # 如果没有命令行参数，交互式输入
        mask_path = input("请输入掩码图片路径: ").strip('"').strip("'")
    else:
        mask_path = sys.argv[1]

    check_mask_values(mask_path)


if __name__ == "__main__":
    main()