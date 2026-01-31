import json
import sys
import os

def transfer_label(json_path, old_label, new_label):
    # 1. 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 错误: 找不到文件 '{json_path}'")
        return

    try:
        # 2. 读取 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        # 3. 遍历并修改 label
        for shape in data.get('shapes', []):
            if str(shape.get('label')) == str(old_label):
                shape['label'] = str(new_label)
                count += 1

        # 4. 如果有修改，则保存文件
        if count > 0:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ 成功完成! 已将 {count} 个标签从 '{old_label}' 修改为 '{new_label}'。")
        else:
            print(f"ℹ️ 未发现标签为 '{old_label}' 的目标。")

    except Exception as e:
        print(f"❌ 处理过程中出现异常: {e}")

if __name__ == "__main__":
    # 检查参数数量是否正确
    if len(sys.argv) != 4:
        print("💡 使用方法: python transfer_labelme.py <json路径> <原标签> <新标签>")
        print('示例: python transfer_labelme.py "XXX.json" 1 2')
    else:
        path = sys.argv[1]
        old = sys.argv[2]
        new = sys.argv[3]
        transfer_label(path, old, new)