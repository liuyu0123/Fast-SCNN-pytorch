import os
import argparse
import torch
import torch.utils.data as data

from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from data_loader.water_val import CustomSegmentationDataset  # 导入自定义数据集
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete


def parse_args():
    """Evaluation Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN Evaluation')
    
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='water',
                        help='dataset name (default: water)')
    
    # ========== 新增：直接指定路径的参数 ==========
    parser.add_argument('--test-images', type=str, default=None,
                        help='测试图像文件夹路径')
    parser.add_argument('--test-masks', type=str, default=None,
                        help='测试掩码文件夹路径（可选，用于计算指标）')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='分割类别数（包括背景）')
    # ============================================
    
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--outdir', default='test_result',
                        help='output directory for visualization')
    parser.add_argument('--save-mask', action='store_true', default=False,
                        help='save predicted masks')
    parser.add_argument('--save-overlay', action='store_true', default=False,
                        help='save overlay visualization')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # 检查必要参数
    if args.test_images is None:
        parser.error("--test-images 参数必须指定")
    if args.model_path is None:
        parser.error("--model-path 参数必须指定（模型权重文件路径）")
    
    print(args)
    return args


class TestDataset(Dataset):
    """
    仅测试用的数据集（无标签）
    """
    def __init__(self, images_dir, transform=None, crop_size=768):
        self.images_dir = images_dir
        self.crop_size = crop_size
        self.transform = transform
        
        # 获取所有图像文件
        self.images = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        
        print(f"[test] 找到 {len(self.images)} 个测试样本")
        print(f"  图像路径: {images_dir}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Resize到crop_size
        img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        
        # 应用transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.images[idx]  # 返回图像和文件名

    def __len__(self):
        return len(self.images)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        
        # 创建输出目录
        self.outdir = args.outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            
        # 创建子目录
        self.mask_dir = os.path.join(self.outdir, 'masks')
        self.overlay_dir = os.path.join(self.outdir, 'overlays')
        if args.save_mask and not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
        if args.save_overlay and not os.path.exists(self.overlay_dir):
            os.makedirs(self.overlay_dir)
        
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # 根据是否有标签选择数据集
        if args.test_masks is not None:
            # 有标签，使用CustomSegmentationDataset计算指标
            self.has_labels = True
            test_dataset = CustomSegmentationDataset(
                images_dir=args.test_images,
                masks_dir=args.test_masks,
                transform=input_transform,
                crop_size=args.crop_size,
                mode='val',  # 使用val模式，不做数据增强
                num_classes=args.num_classes
            )
            self.test_loader = data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True)
            self.metric = SegmentationMetric(args.num_classes)
            self.filenames = None
        else:
            # 无标签，仅推理
            self.has_labels = False
            test_dataset = TestDataset(
                images_dir=args.test_images,
                transform=input_transform,
                crop_size=args.crop_size
            )
            self.test_loader = data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True)
            self.metric = None
            self.filenames = test_dataset.images
        
        # 创建网络
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=False)
        
        # 加载模型权重
        if os.path.isfile(args.model_path):
            print(f'Loading model from {args.model_path}...')
            state_dict = torch.load(args.model_path, map_location='cpu')
            # 处理DataParallel保存的模型
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            print('Finished loading model!')
        else:
            raise FileNotFoundError(f"找不到模型文件: {args.model_path}")
        
        self.model.to(args.device)
        self.model.eval()

    def eval(self):
        if self.has_labels:
            self._eval_with_labels()
        else:
            self._eval_without_labels()
    
    def _eval_with_labels(self):
        """有标签的评估，计算指标"""
        self.metric.reset()
        total_loss = 0.0
        
        with torch.no_grad():
            for i, (image, label) in enumerate(self.test_loader):
                image = image.to(self.args.device)
                label = label.to(self.args.device)
                
                outputs = self.model(image)
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                label_np = label.cpu().numpy()
                
                self.metric.update(pred, label_np)
                pixAcc, mIoU = self.metric.get()
                
                print('Sample %d/%d, pixAcc: %.3f%%, mIoU: %.3f%%' % 
                      (i + 1, len(self.test_loader), pixAcc * 100, mIoU * 100))
                
                # 保存可视化结果
                self._save_visualization(pred.squeeze(0), i, image.cpu())
        
        # 最终指标
        final_pixAcc, final_mIoU = self.metric.get()
        print('\n' + '='*50)
        print('Final Results:')
        print(f'  Pixel Accuracy: {final_pixAcc*100:.3f}%')
        print(f'  mIoU: {final_mIoU*100:.3f}%')
        print('='*50)
    
    def _eval_without_labels(self):
        with torch.no_grad():
            for i, (image, filename) in enumerate(self.test_loader):
                image = image.to(self.args.device)
                
                outputs = self.model(image)
                pred = torch.argmax(outputs[0], 1)
                pred_np = pred.cpu().data.numpy().squeeze(0)
                
                # 解包 filename（DataLoader 会将其打包为列表）
                if isinstance(filename, (list, tuple)):
                    filename = filename[0]
                
                print(f'Processing {i+1}/{len(self.test_loader)}: {filename}')
                self._save_visualization(pred_np, i, image.cpu(), filename)
    
    def _save_visualization(self, pred, idx, image=None, filename=None):
        if not self.args.save_mask and not self.args.save_overlay:
            return
            
        if filename is None:
            filename = f'seg_{idx}.png'
        else:
            filename = os.path.splitext(filename)[0] + '.png'
        
        mask_color = get_color_pallete(pred, self.args.dataset)
        
        if self.args.save_mask:
            mask_path = os.path.join(self.mask_dir, filename)
            mask_color.save(mask_path)
        
        if self.args.save_overlay and image is not None:
            overlay = self._create_overlay(image, pred, mask_color)
            overlay_path = os.path.join(self.overlay_dir, filename)
            overlay.save(overlay_path)
    
    def _create_overlay(self, image, pred, mask_color, alpha=0.6):
        mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
        std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
        img = image.squeeze(0) * std + mean
        img = torch.clamp(img, 0, 1)
        img_np = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        if img_pil.size != mask_color.size:
            mask_color = mask_color.resize(img_pil.size, Image.NEAREST)
        
        # 转换 mask_color 为 RGB 模式
        if mask_color.mode != 'RGB':
            mask_color = mask_color.convert('RGB')
        
        return Image.blend(img_pil, mask_color, alpha)


if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print(f'Testing model: {args.model_path}')
    print(f'Test images: {args.test_images}')
    if args.test_masks:
        print(f'Test masks: {args.test_masks}')
    evaluator.eval()