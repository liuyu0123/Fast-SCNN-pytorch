import json
import os
import argparse
from PIL import Image, ImageDraw
from pathlib import Path
import glob


# python make_gt_red.py \
#     --input-dir "D:\Files\Data_Personal\WelaBoat\WaterSegResize\masks_json" \
#     --output-dir "D:\Files\Data_Personal\WelaBoat\WaterSegResize\masks_red"

def json_to_mask(json_path, output_path, mask_color=(255, 0, 0), bg_color=(0, 0, 0)):
    """
    将单个 JSON 文件转换为红黑 Mask 图像
    
    Args:
        json_path: JSON 文件路径
        output_path: 输出图片路径
        mask_color: 标注区域颜色，默认红色 (255, 0, 0)
        bg_color: 背景颜色，默认黑色 (0, 0, 0)
    """
    try:
        # 读取 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        height = data.get('imageHeight', 320)
        width = data.get('imageWidth', 640)
        
        # 创建黑色背景图像 (RGB模式)
        mask = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(mask)
        
        # 遍历所有标注形状
        shapes = data.get('shapes', [])
        
        if not shapes:
            print(f"警告: {json_path} 中没有找到标注形状")
        
        for shape in shapes:
            shape_type = shape.get('shape_type', 'polygon')
            points = shape.get('points', [])
            label = shape.get('label', 'unknown')
            
            if not points:
                continue
            
            # 将坐标转换为元组列表 [(x1,y1), (x2,y2), ...]
            polygon_points = [(float(x), float(y)) for x, y in points]
            
            # 根据形状类型绘制
            if shape_type == 'polygon':
                # 填充多边形（红色）
                draw.polygon(polygon_points, fill=mask_color, outline=mask_color)
            elif shape_type == 'rectangle':
                # 处理矩形（转换为左上角和右下角坐标）
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    draw.rectangle([(x1, y1), (x2, y2)], fill=mask_color, outline=mask_color)
            elif shape_type == 'circle':
                # 处理圆形
                if len(points) == 2:
                    center = points[0]
                    edge = points[1]
                    radius = ((center[0]-edge[0])**2 + (center[1]-edge[1])**2) ** 0.5
                    x, y = center
                    draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], 
                               fill=mask_color, outline=mask_color)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 保存图片
        mask.save(output_path, 'PNG')
        print(f"✓ 已生成: {output_path} ({len(shapes)} 个标注)")
        
    except json.JSONDecodeError:
        print(f"✗ 错误: {json_path} 不是有效的 JSON 文件")
    except Exception as e:
        print(f"✗ 错误处理 {json_path}: {str(e)}")


def batch_convert(input_dir, output_dir, mask_color=(255, 0, 0), bg_color=(0, 0, 0)):
    """
    批量转换文件夹中的所有 JSON 文件
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 {input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"在 {input_dir} 中没有找到 .json 文件")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件，开始转换...")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    success_count = 0
    
    for json_path in json_files:
        # 生成输出文件名（将 .json 替换为 .png）
        base_name = Path(json_path).stem
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        json_to_mask(json_path, output_path, mask_color, bg_color)
        success_count += 1
    
    print("-" * 50)
    print(f"转换完成: {success_count}/{len(json_files)} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description='将 LabelMe JSON 标注文件转换为红黑 Mask 图像',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个文件
  python json_to_mask.py -i data/1.json -o mask.png
  
  # 批量转换
  python json_to_mask.py -id ./jsons -od ./masks
  
  # 自定义颜色（蓝色标注，白色背景）
  python json_to_mask.py -id ./jsons -od ./masks --mask-color "0,0,255" --bg-color "255,255,255"
        """
    )
    
    # 参数设置
    parser.add_argument('-i', '--input', help='单个 JSON 文件路径')
    parser.add_argument('-o', '--output', help='输出图片路径（单个文件模式）')
    parser.add_argument('-id', '--input-dir', help='输入文件夹路径（批量模式）')
    parser.add_argument('-od', '--output-dir', default='./masks', help='输出文件夹路径（默认: ./masks）')
    parser.add_argument('--mask-color', default='255,0,0', 
                       help='标注颜色，RGB格式如 "255,0,0"（默认红色）')
    parser.add_argument('--bg-color', default='0,0,0', 
                       help='背景颜色，RGB格式如 "0,0,0"（默认黑色）')
    
    args = parser.parse_args()
    
    # 解析颜色
    mask_color = tuple(map(int, args.mask_color.split(',')))
    bg_color = tuple(map(int, args.bg_color.split(',')))
    
    # 验证参数
    if args.input and args.output:
        # 单文件模式
        json_to_mask(args.input, args.output, mask_color, bg_color)
    elif args.input_dir:
        # 批量模式
        batch_convert(args.input_dir, args.output_dir, mask_color, bg_color)
    else:
        parser.print_help()
        print("\n错误: 请指定 --input 和 --output（单文件），或 --input-dir（批量）")


if __name__ == "__main__":
    main()