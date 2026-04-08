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
    
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='water',
                        help='dataset name (default: water)')
    parser.add_argument('--input', type=str, required=True,
                        help='输入路径：单张图片或图片文件夹')
    parser.add_argument('--mask', type=str, default=None,
                        help='真值mask路径：单张mask或mask文件夹（可选，用于计算指标）')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件夹路径（保存叠加图和CSV结果）')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='分割类别数（包括背景）')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if not os.path.exists(args.input):
        parser.error(f"输入路径不存在: {args.input}")
    if args.mask is not None and not os.path.exists(args.mask):
        parser.error(f"Mask路径不存在: {args.mask}")
    
    args.is_single_image = os.path.isfile(args.input)
    if args.is_single_image:
        print(f"检测到单张图片输入: {args.input}")
    else:
        print(f"检测到文件夹输入: {args.input}")
    
    if args.mask is not None:
        args.mask_is_single = os.path.isfile(args.mask)
        if args.is_single_image != args.mask_is_single:
            parser.error("输入图片和mask必须同时为单张文件或同时为文件夹")
    
    print(args)
    return args


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def save_results_to_csv(per_image_metrics, overall_metrics, model_info, args):
    if args.output is None:
        return
    save_path = Path(args.output) / "evaluation_results.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Evaluation Results'])
        writer.writerow([])
        writer.writerow(['Model Information'])
        writer.writerow(['Model Path', args.model_path])
        writer.writerow(['Model Type', args.model])
        writer.writerow(['Total Parameters', f"{model_info['total_params']:,}"])
        writer.writerow(['Model Size (MB)', f"{model_info['model_size_mb']:.2f}"])
        writer.writerow(['Number of Classes', args.num_classes])
        writer.writerow(['Crop Size', args.crop_size])
        writer.writerow([])
        writer.writerow(['Per-Image Results'])
        writer.writerow(['Image Name', 'Precision', 'Recall', 'F1-Score', 'mIoU', 'Accuracy'])
        for item in per_image_metrics:
            writer.writerow([
                item['filename'],
                f"{item['precision']:.6f}",
                f"{item['recall']:.6f}",
                f"{item['f1']:.6f}",
                f"{item['miou']:.6f}",
                f"{item['acc']:.6f}"
            ])
        writer.writerow([])
        writer.writerow(['Overall Results'])
        writer.writerow(['Total Images', overall_metrics.get('total_images', 0)])
        writer.writerow(['Average Precision', f"{overall_metrics.get('precision', 0):.6f}"])
        writer.writerow(['Average Recall', f"{overall_metrics.get('recall', 0):.6f}"])
        writer.writerow(['Average F1-Score', f"{overall_metrics.get('f1', 0):.6f}"])
        writer.writerow(['Average mIoU', f"{overall_metrics.get('miou', 0):.6f}"])
        writer.writerow(['Average Accuracy', f"{overall_metrics.get('acc', 0):.6f}"])
        writer.writerow(['Average Inference Time (ms)', f"{overall_metrics.get('inference_time_ms', 0):.4f}"])
        writer.writerow(['FPS', f"{overall_metrics.get('fps', 0):.2f}"])
    print(f"\nCSV Results saved to {save_path}")


class AdaptiveDataset(data.Dataset):
    def __init__(self, input_path, mask_path=None, transform=None, crop_size=768):
        self.crop_size = crop_size
        self.transform = transform
        self.mask_path = mask_path
        
        if os.path.isfile(input_path):
            self.images = [input_path]
            if mask_path is not None:
                if os.path.isfile(mask_path):
                    self.masks = [mask_path]
                else:
                    raise ValueError("输入为单张图片时，mask也必须为单张图片")
            else:
                self.masks = None
            self.is_single = True
        else:
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            self.images = sorted([
                os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.lower().endswith(valid_exts)
            ])
            if mask_path is not None:
                self.masks = sorted([
                    os.path.join(mask_path, f) for f in os.listdir(mask_path)
                    if f.lower().endswith(valid_exts)
                ])
                if len(self.images) != len(self.masks):
                    raise ValueError(f"图片数量({len(self.images)})与mask数量({len(self.masks)})不匹配")
            else:
                self.masks = None
            self.is_single = False
        print(f"[Dataset] 加载了 {len(self.images)} 个样本")

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size
        
        img_resized = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        if self.transform is not None:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)
        
        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
            mask_np = np.array(mask)
            mask_np = (mask_np > 0).astype(np.int64)
            mask_tensor = torch.from_numpy(mask_np)
            filename = os.path.basename(img_path)
            return img_tensor, mask_tensor, filename, img_path, orig_size
        else:
            filename = os.path.basename(img_path)
            return img_tensor, filename, img_path, orig_size

    def __len__(self):
        return len(self.images)


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        
        if args.output:
            self.output_dir = Path(args.output)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.overlay_dir = self.output_dir / "overlays"
            self.overlay_dir.mkdir(exist_ok=True)
            print(f"结果将保存至: {self.output_dir}")
        else:
            self.output_dir = None
            self.overlay_dir = None
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        self.has_labels = args.mask is not None
        test_dataset = AdaptiveDataset(
            input_path=args.input,
            mask_path=args.mask if self.has_labels else None,
            transform=input_transform,
            crop_size=args.crop_size
        )
        
        self.test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        if self.has_labels:
            self.metric = SegmentationMetric(args.num_classes)
        
        self.model = get_fast_scnn(args.dataset, aux=args.aux, pretrained=False)
        self.model_info = get_model_info(self.model)
        print(f"Model: {self.model_info['total_params']:,} params, {self.model_info['model_size_mb']:.2f} MB")
        
        if os.path.isfile(args.model_path):
            print(f'Loading model from {args.model_path}...')
            state_dict = torch.load(args.model_path, map_location='cpu')
            new_state_dict = {k.replace('module.', '') if k.startswith('module.') else k: v 
                            for k, v in state_dict.items()}
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
            return None, None

    def _eval_with_labels(self):
        per_image_results = []
        inference_times = []
        total_images = 0
        test_start = time.time()
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                image, label, filename, img_path, orig_size = batch
                image = image.to(self.args.device)
                label = label.to(self.args.device)
                
                filename = filename[0] if isinstance(filename, (list, tuple)) else filename
                img_path = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
                
                # 处理 orig_size (可能是tensor或list)
                if isinstance(orig_size, torch.Tensor):
                    orig_size = (int(orig_size[0].item()), int(orig_size[1].item()))
                elif isinstance(orig_size, (list, tuple)):
                    if isinstance(orig_size[0], torch.Tensor):
                        orig_size = (int(orig_size[0].item()), int(orig_size[1].item()))
                    else:
                        orig_size = (int(orig_size[0]), int(orig_size[1]))
                
                if self.args.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_start = time.time()
                outputs = self.model(image)
                if self.args.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_time = time.time() - batch_start
                inference_times.append(batch_time)
                
                pred = torch.argmax(outputs[0], 1)
                pred_np = pred.cpu().data.numpy()
                label_np = label.cpu().numpy()
                
                self.metric.reset()
                self.metric.update(pred_np, label_np)
                metrics = self.metric.get_full_metrics()
                
                per_image_results.append({
                    'filename': filename,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'miou': metrics['mIoU'],
                    'acc': metrics['pixAcc'],
                    'inference_time_ms': batch_time * 1000
                })
                
                if self.output_dir:
                    self._save_overlay(pred_np.squeeze(0), img_path, orig_size, filename)
                total_images += 1
        
        test_time = time.time() - test_start
        avg_inference_time_ms = np.mean(inference_times) * 1000 if inference_times else 0
        fps = total_images / test_time if test_time > 0 else 0
        
        overall_metrics = {
            'total_images': total_images,
            'precision': np.mean([r['precision'] for r in per_image_results]),
            'recall': np.mean([r['recall'] for r in per_image_results]),
            'f1': np.mean([r['f1'] for r in per_image_results]),
            'miou': np.mean([r['miou'] for r in per_image_results]),
            'acc': np.mean([r['acc'] for r in per_image_results]),
            'inference_time_ms': avg_inference_time_ms,
            'fps': fps
        }
        
        print('\n' + '='*50)
        print('Evaluation Summary')
        print('='*50)
        print(f'Total Images: {total_images}')
        print(f'Average Precision: {overall_metrics["precision"]:.6f}')
        print(f'Average Recall:    {overall_metrics["recall"]:.6f}')
        print(f'Average F1-Score:  {overall_metrics["f1"]:.6f}')
        print(f'Average mIoU:      {overall_metrics["miou"]:.6f}')
        print(f'Average Accuracy:  {overall_metrics["acc"]:.6f}')
        print(f'Inference Time:    {avg_inference_time_ms:.4f} ms')
        print(f'FPS:               {fps:.2f}')
        print('='*50)
        
        return per_image_results, overall_metrics

    def _eval_without_labels(self):
        with torch.no_grad():
            for batch in self.test_loader:
                image, filename, img_path, orig_size = batch
                image = image.to(self.args.device)
                
                filename = filename[0] if isinstance(filename, (list, tuple)) else filename
                img_path = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
                
                if isinstance(orig_size, torch.Tensor):
                    orig_size = (int(orig_size[0].item()), int(orig_size[1].item()))
                elif isinstance(orig_size, (list, tuple)):
                    if isinstance(orig_size[0], torch.Tensor):
                        orig_size = (int(orig_size[0].item()), int(orig_size[1].item()))
                    else:
                        orig_size = (int(orig_size[0]), int(orig_size[1]))
                
                outputs = self.model(image)
                pred = torch.argmax(outputs[0], 1)
                pred_np = pred.cpu().data.numpy().squeeze(0)
                
                if self.output_dir:
                    self._save_overlay(pred_np, img_path, orig_size, filename)

    def _save_overlay(self, pred, img_path, orig_size, filename):
        if not self.output_dir:
            return
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        base_name = Path(filename).stem + '.png'
        
        orig_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = orig_size
        
        if orig_img.size != (orig_w, orig_h):
            orig_img = orig_img.resize((orig_w, orig_h), Image.BILINEAR)
        
        pred_img = Image.fromarray((pred * 255).astype(np.uint8))
        pred_resized = pred_img.resize((orig_w, orig_h), Image.NEAREST)
        pred_array = np.array(pred_resized) // 255
        
        overlay_array = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        overlay_array[pred_array == 1, 0] = 255
        overlay_array[pred_array == 1, 3] = 128
        
        overlay_img = Image.fromarray(overlay_array, mode='RGBA')
        if orig_img.mode != 'RGBA':
            orig_img = orig_img.convert('RGBA')
        result = Image.alpha_composite(orig_img, overlay_img)
        
        output_path = self.overlay_dir / base_name
        result.save(output_path)


if __name__ == '__main__':
    args = parse_args()
    evaluator = Evaluator(args)
    print(f'Testing model: {args.model_path}')
    print(f'Input: {args.input}')
    if args.mask:
        print(f'Mask: {args.mask}')
    
    per_image_metrics, overall_metrics = evaluator.eval()
    
    if per_image_metrics is not None and overall_metrics is not None:
        if args.output:
            save_results_to_csv(per_image_metrics, overall_metrics, evaluator.model_info, args)
        else:
            print("\n提示：使用 --output 参数指定输出路径可保存CSV结果")
    elif args.output:
        print(f"\n叠加图已保存至: {args.output}/overlays/")