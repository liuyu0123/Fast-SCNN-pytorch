import os
import json
import numpy as np
import cv2
import argparse

# 调用：
# python make_gt_pspnet.py \
#     --input "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_json" \
#     --output "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_pspnet"

def create_pspnet_labels(json_folder, output_folder):
    """
    将 LabelMe JSON 文件转换为 PSPNet 适用的单通道灰度 Label 图。
    
    参数:
        json_folder: 存放 .json 文件的文件夹路径
        output_folder: 输出 .png label 图的文件夹路径
    """
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 用于收集所有出现的类别名称，以便生成 class_names.txt
    all_labels = set()
    
    # 获取所有 json 文件
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ 在 {json_folder} 中未找到任何 .json 文件。")
        return

    print(f"🚀 开始处理 {len(json_files)} 个标注文件...")

    for i, json_file in enumerate(json_files):
        json_path = os.path.join(json_folder, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. 获取图像尺寸
            height = data['imageHeight']
            width = data['imageWidth']
            
            # 2. 创建全黑的空白掩码 (背景默认为 0)
            # dtype=np.uint8 是关键，PSPNet 需要单通道 8位整数
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 3. 处理 shapes
            shapes = data.get('shapes', [])
            
            for shape in shapes:
                label = shape['label']
                points = np.array(shape['points'], dtype=np.int32) # 坐标取整
                
                # 收集类别
                all_labels.add(label)
                
                # 注意：这里暂时不填色，因为我们需要先确定所有类别的 ID 映射
                # 为了简单起见，我们采用动态映射策略：
                # 但为了效率，通常建议先定义好映射。
                # 此处为了演示通用性，我们先暂存形状信息，稍后统一映射，
                # 或者更简单的做法：如果类别不多，直接在循环外定义映射。
                # 
                # 【优化策略】：由于每个 JSON 可能包含不同子集，
                # 最稳健的方法是：先扫描所有文件确定全局类别表，再第二次循环生成图片。
                # 但为了脚本简洁，我们假设用户希望按字母顺序自动分配 ID。
                # 我们将在本循环中先不做 fillPoly，而是收集数据。
                pass

            # --- 重新设计逻辑以支持动态类别映射 ---
            # 为了单次遍历完成，我们需要预先知道 label -> id 的映射。
            # 既然我们无法在一次遍历中既知道全局类别又画图（除非跑两遍），
            # 这里采用一种常用策略：
            # 1. 先收集当前文件的标签。
            # 2. 如果是小数据集，建议手动指定映射。
            # 3. 如果是自动模式，我们这里做一个妥协：
            #    我们先生成临时的映射关系用于当前文件，但这可能导致不同文件间 ID 不一致！
            #    ❌ 错误做法：每个文件独立分配 ID (会导致训练灾难)
            #    ✅ 正确做法：必须有一个全局统一的 label_map
            
            # 因此，我们将逻辑分为两步：
            # 第一步：收集所有标签（上面已做）
            # 第二步：生成图片（下面做）
            
        except Exception as e:
            print(f"⚠️ 读取 {json_file} 失败: {e}")
            continue

    # === 第二步：建立全局类别映射并生成图片 ===
    
    # 排序类别，确保 ID 固定 (背景通常单独处理，这里假设 0 是背景)
    # 如果数据集中有 "background" 标签，通常我们会跳过它或将其设为 0
    sorted_labels = sorted(list(all_labels))
    
    # 构建映射字典: label -> id
    # 假设 0 永远是背景 (Background)，即使 JSON 里没有显式的 background 多边形
    # 其他标签从 1 开始编号
    label_map = {}
    current_id = 1
    for label in sorted_labels:
        # 如果标注里明确写了 background，可以特殊处理，通常忽略或设为0
        if label.lower() == 'background':
            label_map[label] = 0
        else:
            label_map[label] = current_id
            current_id += 1
            
    print(f"📋 检测到以下类别及对应 ID:")
    for l, id_val in label_map.items():
        print(f"   {l}: {id_val}")
    
    # 保存类别映射文件，方便配置训练参数
    map_file_path = os.path.join(output_folder, 'class_names.txt')
    with open(map_file_path, 'w') as f:
        for label in sorted_labels:
            if label.lower() != 'background': # 只记录前景类别，或者按需调整
                f.write(f"{label_map[label]} {label}\n")
    print(f"💾 类别映射已保存至: {map_file_path}")

    # 再次遍历生成图片
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            height = data['imageHeight']
            width = data['imageWidth']
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for shape in data['shapes']:
                label = shape['label']
                points = np.array(shape['points'], dtype=np.int32)
                
                if label in label_map:
                    class_id = label_map[label]
                    # 填充多边形
                    # cv2.fillPoly 接受 [points] 列表
                    cv2.fillPoly(mask, [points], color=class_id)
                else:
                    print(f"⚠️ 未知标签 {label} 在 {json_file} 中被跳过")
            
            # 生成输出文件名：去掉 .json 后缀，加上 _label.png 或直接同名
            # PSPNet 通常要求 img 和 label 同名，放在不同文件夹
            base_name = os.path.splitext(json_file)[0]
            output_path = os.path.join(output_folder, f"{base_name}.png")
            
            cv2.imwrite(output_path, mask)
            
            # 简单验证
            unique_vals = np.unique(mask)
            # print(f"✅ 生成成功: {output_path} (包含类别ID: {unique_vals})")
            
        except Exception as e:
            print(f"❌ 处理 {json_file} 时出错: {e}")

    print(f"\n🎉 完成！所有 Label 图已保存至: {output_folder}")
    print(f"💡 提示：请检查生成的图片是否为单通道 (Mode: L)。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON to PSPNet Ground Truth')
    parser.add_argument('--input', type=str, default='./json_annotations', help='输入 JSON 文件夹路径')
    parser.add_argument('--output', type=str, default='./gt_labels', help='输出 PNG Label 文件夹路径')
    
    args = parser.parse_args()
    
    # 如果默认路径不存在，尝试使用当前目录下的常见命名
    if not os.path.exists(args.input):
        # 尝试查找当前目录下的 json 文件
        current_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if current_files:
            print("⚠️ 未指定输入文件夹，但发现当前目录有 JSON 文件。")
            print("   用法示例: python make_gt_pspnet.py --input ./jsons --output ./labels")
            # 为了演示，如果不传参且没默认文件夹，我们可以创建一个演示用的逻辑或直接退出提示
            # 这里我们直接提示用户
            print(f"   请将 JSON 文件放入 '{args.input}' 文件夹，或运行命令指定路径。")
            exit(0)
            
    create_pspnet_labels(args.input, args.output)