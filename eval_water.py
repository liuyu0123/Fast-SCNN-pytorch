import os
import csv
import argparse
import time
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms
from PIL import Image

from data_loader.water_val import CustomSegmentationDataset
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
    
    # 路径参数
    parser.add_argument('--test-images', type=str, default=None,
                        help='测试图像文件夹路径')
    parser.add_argument('--test-masks', type=str, default=None,
                        help='测试掩码文件夹路径（可选，用于计算指标）')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='分割类别数（包括背景）')
    
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
    # 新增：输出结果路径
    parser.add_argument('--output', type=str, default=None,
                        help='测试结果CSV保存路径（可选）')
    
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


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def save_results_to_csv(metrics, model_info, args, save_path):
    """保存测试结果到 CSV（标准格式）"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建结果字典
    result = {
        'model_path': args.model_path,
        'model_type': args.model,
        'test_images': args.test_images,
        'test_masks': args.test_masks if args.test_masks else 'N/A',
        'crop_size': args.crop_size,
        'num_classes': args.num_classes,
        'total_params': model_info['total_params'],
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
        'test_loss': f"{metrics.get('loss', 0):.6f}",
        'test_precision': f"{metrics.get('precision', 0):.6f}",
        'test_recall': f"{metrics.get('recall', 0):.6f}",
        'test_f1': f"{metrics.get('f1', 0):.6f}",
        'test_miou': f"{metrics.get('miou', 0):.6f}",
        'test_acc': f"{metrics.get('acc', 0):.6f}",
        'inference_time_ms': f"{metrics.get('inference_time_ms', 0):.4f}",
        'fps': f"{metrics.get('fps', 0):.2f}",
        'total_images': metrics.get('total_images', 0),
    }
    
    # 写入 CSV（追加模式）
    header = list(result.keys())
    file_exists = save_path.exists()
    
    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    
    print(f"Results saved to {save_path}")
    
    # 同时保存详细文本报告
    report_path = save_path.parent / f"{save_path.stem}_report.txt"
    with open(report_path, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Model Type: {args.model}\n")
        f.write(f"Test Images: {args.test_images}\n")
        f.write(f"Test Masks: {args.test_masks if args.test_masks else 'N/A'}\n\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Crop Size: {args.crop_size}\n")
        f.write(f"Number of Classes: {args.num_classes}\n\n")
        f.write(f"Test Set Size: {metrics.get('total_images', 0)} images\n\n")
        f.write(f"Pixel Accuracy: {metrics.get('acc', 0)*100:.3f}%\n")
        f.write(f"mIoU:           {metrics.get('miou', 0):.6f}\n")
        f.write(f"Precision:      {metrics.get('precision', 0):.6f}\n")
        f.write(f"Recall:         {metrics.get('recall', 0):.6f}\n")
        f.write(f"F1-Score:       {metrics.get('f1', 0):.6f}\n")
        f.write(f"Inference Time: {metrics.get('inference_time_ms', 0):.4f} ms\n")
        f.write(f"FPS:            {metrics.get('fps', 0):.2f}\n")
        f.write(f"{'='*50}\n")
    
    print(f"Text report appended to {report_path}")


class TestDataset(data.Dataset):
    """
    仅测试用的数据集（无标签）
    """
    def __init__(self, images_dir, transform=None, crop_size=768):
        self.images_dir = images_dir
        self.crop_size = crop_size
        self.transform = transform
        
        self.images = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        
        print(f"[test] 找到 {len(self.images)} 个测试样本")
        print(f"  图像路径: {images_dir}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        
        img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.images[idx]

    def __len__(self):
        return len(self.images)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        
        # 创建输出目录
        self.outdir = args.outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            
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
            self.has_labels = True
            test_dataset = CustomSegmentationDataset(
                images_dir=args.test_images,
                masks_dir=args.test_masks,
                transform=input_transform,
                crop_size=args.crop_size,
                mode='val',
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
        
        # 创建网络并加载权重
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=False)
        
        # 获取模型信息
        self.model_info = get_model_info(self.model)
        print(f"Model: {self.model_info['total_params']:,} params, {self.model_info['model_size_mb']:.2f} MB")
        
        if os.path.isfile(args.model_path):
            print(f'Loading model from {args.model_path}...')
            state_dict = torch.load(args.model_path, map_location='cpu')
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
            return self._eval_with_labels()
        else:
            self._eval_without_labels()
            return None

    def _eval_with_labels(self):
        """有标签的评估，计算指标"""
        self.metric.reset()
        
        # 测量推理时间
        inference_times = []
        total_images = 0
        
        test_start = time.time()
        
        with torch.no_grad():
            for i, (image, label) in enumerate(self.test_loader):
                image = image.to(self.args.device)
                label = label.to(self.args.device)
                
                # 测量推理时间
                if self.args.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_start = time.time()
                
                outputs = self.model(image)
                
                if self.args.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_time = time.time() - batch_start
                inference_times.append(batch_time)
                
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                label_np = label.cpu().numpy()
                
                self.metric.update(pred, label_np)
                
                # 获取当前指标用于显示（使用新的 get_full_metrics）
                current_metrics = self.metric.get_full_metrics()
                print('Sample %d/%d, pixAcc: %.3f%%, mIoU: %.3f%%, Precision: %.3f%%, Recall: %.3f%%' % 
                      (i + 1, len(self.test_loader), 
                       current_metrics['pixAcc'] * 100, 
                       current_metrics['mIoU'] * 100,
                       current_metrics['precision'] * 100,
                       current_metrics['recall'] * 100))
                
                self._save_visualization(pred.squeeze(0), i, image.cpu())
                
                total_images += image.size(0)
        
        test_time = time.time() - test_start
        
        # 使用 get_full_metrics 获取所有最终指标
        final_metrics = self.metric.get_full_metrics()
        
        # 计算推理时间
        avg_inference_time_ms = np.mean(inference_times) * 1000
        fps = total_images / test_time if test_time > 0 else 0
        
        # 构建指标字典（兼容原有格式）
        metrics = {
            'loss': 0,  # Fast-SCNN eval没有计算loss
            'acc': final_metrics['pixAcc'],
            'miou': final_metrics['mIoU'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'f1': final_metrics['f1'],
            'inference_time_ms': avg_inference_time_ms,
            'fps': fps,
            'total_images': total_images,
        }
        
        print('\n' + '='*50)
        print('Final Results:')
        print(f'  Pixel Accuracy: {final_metrics["pixAcc"]*100:.3f}%')
        print(f'  mIoU:           {final_metrics["mIoU"]:.6f}')
        print(f'  Precision:      {final_metrics["precision"]:.6f}')
        print(f'  Recall:         {final_metrics["recall"]:.6f}')
        print(f'  F1-Score:       {final_metrics["f1"]:.6f}')
        print(f'  Inference Time: {avg_inference_time_ms:.4f} ms')
        print(f'  FPS:            {fps:.2f}')
        print('='*50)
        
        return metrics

    def _eval_without_labels(self):
        """无标签的推理"""
        with torch.no_grad():
            for i, (image, filename) in enumerate(self.test_loader):
                image = image.to(self.args.device)
                
                outputs = self.model(image)
                pred = torch.argmax(outputs[0], 1)
                pred_np = pred.cpu().data.numpy().squeeze(0)
                
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
    
    # 运行评估
    metrics = evaluator.eval()
    
    # 保存结果（如果有标签）
    if metrics is not None:
        if args.output:
            output_path = args.output
        else:
            # 默认保存到模型目录
            model_dir = Path(args.model_path).parent
            output_path = model_dir / f"{args.model}_test_results.csv"
        
        save_results_to_csv(metrics, evaluator.model_info, args, output_path)