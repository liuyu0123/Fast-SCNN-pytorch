import json
import cv2
import numpy as np
import os
import glob
from pathlib import Path

mask_suffix = '_mask'
mask_suffix = ''

# ========== 配置路径 ==================
IMG_DIR    = r"D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted"         # 原图目录（用于获取尺寸）
JSON_DIR   = r"D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_json"    # JSON 标注目录
OUTPUT_DIR = r"D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_gif"     # 输出掩码目录
# =====================================

def poly2mask(json_path, H, W):
    """
    将 Labelme JSON 转换为二值掩码
    水域 = 白色 (255)，背景 = 黑色 (0)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    
    # 创建黑色背景
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 填充所有标注的多边形为白色 (255)
    for shape in ann['shapes']:
        label = shape['label'].lower()
        # 支持多种可能的标注名称
        if label in ['water', '水域', '水面', 'water_surface']:
            pts = np.array(shape['points'], np.int32)
            cv2.fillPoly(mask, [pts], color=255)  # 白色填充
    
    return mask

def convert():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 支持的图片格式
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    img_list = []
    for ext in img_extensions:
        img_list.extend(glob.glob(os.path.join(IMG_DIR, ext)))
        img_list.extend(glob.glob(os.path.join(IMG_DIR, ext.upper())))
    
    print(f"找到 {len(img_list)} 张图片")
    
    success_count = 0
    skip_count = 0
    
    for img_p in img_list:
        base = Path(img_p).stem
        json_p = os.path.join(JSON_DIR, base + '.json')
        
        # 检查 JSON 是否存在
        if not os.path.exists(json_p):
            print(f'[跳过] 未找到标注文件: {json_p}')
            skip_count += 1
            continue
        
        # 读取原图获取尺寸
        img = cv2.imread(img_p)
        if img is None:
            print(f'[错误] 无法读取图片: {img_p}')
            skip_count += 1
            continue
        
        H, W = img.shape[:2]
        
        # 生成掩码
        mask = poly2mask(json_p, H, W)
        
        # 保存为 PNG（无损，支持单通道）
        mask_name = base + mask_suffix + '.png'
        mask_path = os.path.join(OUTPUT_DIR, mask_name)
        cv2.imwrite(mask_path, mask)
        
        print(f'[成功] 已保存: {mask_name}')
        success_count += 1
    
    print(f'\n=== 转换完成 ===')
    print(f'成功: {success_count} | 跳过: {skip_count} | 总计: {len(img_list)}')

if __name__ == '__main__':
    convert()