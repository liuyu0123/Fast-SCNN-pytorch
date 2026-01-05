# json2labelIds_v3.py
import json, cv2, numpy as np, os, glob
from pathlib import Path

# ========== 仅需改这里 ==================
IMG_DIR   = r"D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted"            # 原图目录
JSON_DIR  = r"D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_json"       # json 目录
OUTPUT_DIR= r"D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted_labelIds2"   # 你想放 mask 的地方
# =======================================

CLS_ID = {'water': 1}   # 0=背景 1=water

def poly2mask(json_path, H, W):
    with open(json_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    mask = np.zeros((H, W), dtype=np.uint8)
    for shape in ann['shapes']:
        cls_name = shape['label']
        cls_id   = CLS_ID.get(cls_name, 0)
        pts = np.array(shape['points'], np.int32)
        cv2.fillPoly(mask, [pts], color=cls_id)
    return mask

def convert():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_list = glob.glob(os.path.join(IMG_DIR, '*.jpg')) + \
               glob.glob(os.path.join(IMG_DIR, '*.png'))

    for img_p in img_list:
        base = Path(img_p).stem
        json_p = os.path.join(JSON_DIR, base + '.json')
        if not os.path.exists(json_p):
            print(f'[WARN] json missing: {json_p}')
            continue

        img = cv2.imread(img_p)
        if img is None:
            print(f'[WARN] read img fail: {img_p}')
            continue
        H, W = img.shape[:2]

        mask = poly2mask(json_p, H, W)

        # Cityscapes 要求同名字目录 + 同名文件
        # sub_dir = os.path.join(OUTPUT_DIR, base.split('_')[0])  # 例如 aachen
        sub_dir = OUTPUT_DIR   # 不再按前缀分子目录
        os.makedirs(sub_dir, exist_ok=True)

        mask_name = base + '_gtFine_labelIds.png'
        mask_path = os.path.join(sub_dir, mask_name)
        cv2.imwrite(mask_path, mask)
        print(f'[INFO] saved: {mask_path}')

    print('=== 全部转换完成 ===')

if __name__ == '__main__':
    convert()