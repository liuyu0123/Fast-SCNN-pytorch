# data_loader/water.py  二分类 water 数据集（不继承 CitySegmentation）
import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
from torch.utils.data import Dataset

NUM_CLASS = 2
_KEY   = np.array([0, 1])
_MAPPING = np.zeros(256, dtype='int32')
_MAPPING[1] = 1        # 像素值 1 对应 water

def _class_to_index(mask):
    mask = np.array(mask).astype('int32')
    # index = np.digitize(mask.ravel(), _MAPPING, right=True)
    # return _KEY[index].reshape(mask.shape)
    return np.where(mask == 1, 1, 0)

class WaterSegmentation(Dataset):
    NUM_CLASS = 2
    def __init__(self, root='./datasets/water', split='train', mode=None,
                 transform=None, base_size=1024, crop_size=768):
        self.root      = root
        self.split     = split
        self.mode      = mode if mode else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        # 自己扫图
        self.images, self.masks = self._get_pairs(root, split)
        if len(self.images) == 0:
            raise RuntimeError(f'Found 0 images/masks in {root}/{split}')
        print(f'=> Water {split}: {len(self.images)} samples')

    def _get_pairs(self, root, split):
        img_dir  = os.path.join(root, 'images',  split)
        mask_dir = os.path.join(root, 'masks', split)
        imgs, masks = [], []
        for fname in sorted(os.listdir(img_dir)):
            base, ext = os.path.splitext(fname)
            if ext.lower() not in ['.png', '.jpg', '.jpeg']: continue
            img_path = os.path.join(img_dir, fname)
            # msk_path = os.path.join(mask_dir, base + '.png')  # mask 统一 png
            msk_path = os.path.join(mask_dir, base + '_gtFine_labelIds.png')
            if os.path.isfile(msk_path):
                imgs.append(img_path)
                masks.append(msk_path)
            else:
                print(f'[WARN] mask not found: {msk_path}')
        return imgs, masks

    # === 以下与 Cityscapes 保持一致，直接搬过来 ===
    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        short_size = random.randint(int(self.base_size * 0.5),
                                    int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size; oh = int(1.0 * h * ow / w)
        else:
            oh = short_size; ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return np.array(img), _class_to_index(mask)

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            ow = short_size; oh = int(1.0 * h * ow / w)
        else:
            oh = short_size; ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        return np.array(img), _class_to_index(mask)

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return torch.LongTensor(_class_to_index(mask))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:  # test/testval
            img, mask = self._img_transform(img), self._mask_transform(mask)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        return NUM_CLASS