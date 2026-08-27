import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomSegmentationDataset(Dataset):
    """
    自定义分割数据集，支持直接指定图像和掩码路径
    """
    def __init__(self, images_dir, masks_dir, transform=None, 
                 base_size=1024, crop_size=768, mode='train', num_classes=2):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mode = mode
        self.base_size = base_size
        self.crop_size = crop_size
        self.num_class = num_classes  # 直接作为普通属性
        
        # 获取所有图像文件
        self.images = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        
        # 对应的掩码文件（假设同名）
        self.masks = []
        mask_suffix = '_gtFine_labelIds'
        for img_name in self.images:
            base_name = os.path.splitext(img_name)[0]
            # 尝试常见的掩码扩展名
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                mask_path = os.path.join(masks_dir, base_name + mask_suffix + ext)
                if os.path.exists(mask_path):
                    self.masks.append(base_name + mask_suffix + ext)
                    break
            else:
                raise FileNotFoundError(f"找不到对应的掩码文件: {base_name}.*")
        
        self.transform = transform
        
        # 数据增强
        if mode == 'train':
            self.sync_transform = self._sync_transform_train
        else:
            self.sync_transform = self._sync_transform_val
            
        print(f"[{mode}] 找到 {len(self.images)} 个样本")
        print(f"  图像路径: {images_dir}")
        print(f"  掩码路径: {masks_dir}")

    def _sync_transform_train(self, img, mask):
        # 随机缩放
        short_size = np.random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 随机裁剪 - 确保图像足够大
        w, h = img.size
        
        # 如果图像小于crop_size，先resize到crop_size
        if w < self.crop_size or h < self.crop_size:
            img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
            w, h = img.size
        
        # 现在可以安全地裁剪
        x1 = np.random.randint(0, w - self.crop_size + 1)
        y1 = np.random.randint(0, h - self.crop_size + 1)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        
        # 随机水平翻转
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
        return img, mask

    def _sync_transform_val(self, img, mask):
        # 验证时直接resize到crop_size
        img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
        return img, mask

    def __getitem__(self, idx):
        # 加载图像和掩码
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 灰度图
        
        # 同步变换
        img, mask = self.sync_transform(img, mask)
        
        # 应用transform（ToTensor和Normalize）
        if self.transform is not None:
            img = self.transform(img)
            
        # 掩码转为tensor（类别索引）
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask

    def __len__(self):
        return len(self.images)