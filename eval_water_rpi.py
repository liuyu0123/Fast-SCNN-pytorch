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

from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric
from utils.visualize import get_color_pallete


def parse_args():
    """Raspberry Pi 3B Optimized Evaluation Options"""
    parser = argparse.ArgumentParser(description='Fast-SCNN RPi3B Optimized Evaluation')

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

    # RPi optimization parameters
    parser.add_argument('--input-size', type=int, default=256,
                        help='推理输入尺寸（高和宽），RPi3B建议256或更小（默认：256）')
    parser.add_argument('--num-threads', type=int, default=4,
                        help='PyTorch CPU线程数，RPi3B建议设为4（默认：4）')
    parser.add_argument('--use-jit', action='store_true', default=False,
                        help='使用TorchScript JIT trace加速推理')
    parser.add_argument('--warmup-iters', type=int, default=10,
                        help='预热迭代次数，让CPU频率稳定（默认：10）')
    parser.add_argument('--no-overlay-resize', action='store_true', default=False,
                        help='不将预测结果resize回原始尺寸，直接保存低分辨率overlay以节省后处理时间')
    parser.add_argument('--no-save-overlay', action='store_true', default=False,
                        help='不保存overlay图片，仅输出CSV指标（最大化推理速度）')

    args = parser.parse_args()

    # Force CPU on Raspberry Pi
    device = torch.device("cpu")
    args.device = device

    # Set thread count for better RPi performance
    torch.set_num_threads(args.num_threads)

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

    print(f"设备: CPU | 线程数: {args.num_threads} | 输入尺寸: {args.input_size}x{args.input_size}")
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
    save_path = Path(args.output) / "evaluation_results_rpi.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Raspberry Pi 3B Optimized Evaluation Results'])
        writer.writerow([])
        writer.writerow(['Optimization Config'])
        writer.writerow(['Input Size', args.input_size])
        writer.writerow(['Num Threads', args.num_threads])
        writer.writerow(['JIT Enabled', args.use_jit])
        writer.writerow(['Overlay Resize', not args.no_overlay_resize])
        writer.writerow([])
        writer.writerow(['Model Information'])
        writer.writerow(['Model Path', args.model_path])
        writer.writerow(['Model Type', args.model])
        writer.writerow(['Total Parameters', f"{model_info['total_params']:,}"])
        writer.writerow(['Model Size (MB)', f"{model_info['model_size_mb']:.2f}"])
        writer.writerow(['Number of Classes', args.num_classes])
        writer.writerow([])
        writer.writerow(['Per-Image Results'])
        writer.writerow(['Image Name', 'Precision', 'Recall', 'F1-Score', 'mIoU', 'Accuracy', 'Inference Time (ms)', 'FPS'])
        for item in per_image_metrics:
            writer.writerow([
                item['filename'],
                f"{item['precision']:.6f}",
                f"{item['recall']:.6f}",
                f"{item['f1']:.6f}",
                f"{item['miou']:.6f}",
                f"{item['acc']:.6f}",
                f"{item['inference_time_ms']:.4f}",
                f"{item['fps']:.2f}"
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
        writer.writerow(['Pure Model FPS', f"{overall_metrics.get('fps', 0):.2f}"])
        writer.writerow(['End-to-End FPS', f"{overall_metrics.get('end_to_end_fps', 0):.2f}"])
    print(f"\nCSV Results saved to {save_path}")


class RPIDataset(data.Dataset):
    """轻量级数据集，针对RPi CPU推理优化：
    - 不使用多进程DataLoader的复杂transform
    - 直接resize到目标尺寸，避免random crop等训练操作
    """
    def __init__(self, input_path, mask_path=None, transform=None, input_size=256):
        self.input_size = input_size
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
            else:
                self.masks = None
            self.is_single = False
        print(f"[RPIDataset] 加载了 {len(self.images)} 个样本")

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size  # (W, H)

        # 直接resize到目标尺寸，不做crop
        img_resized = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        if self.transform is not None:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = transforms.ToTensor()(img_resized)

        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
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


class RPiEvaluator(object):
    def __init__(self, args):
        self.args = args

        if args.output:
            self.output_dir = Path(args.output)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if not args.no_save_overlay:
                self.overlay_dir = self.output_dir / "overlays"
                self.overlay_dir.mkdir(exist_ok=True)
            else:
                self.overlay_dir = None
            print(f"结果将保存至: {self.output_dir}")
            if args.no_save_overlay:
                print("已启用 --no-save-overlay，仅保存CSV结果，不生成overlay图片")
        else:
            self.output_dir = None
            self.overlay_dir = None

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        self.has_labels = args.mask is not None
        test_dataset = RPIDataset(
            input_path=args.input,
            mask_path=args.mask if self.has_labels else None,
            transform=input_transform,
            input_size=args.input_size
        )

        # RPi优化：num_workers=0避免进程开销，pin_memory=False因为CPU推理不需要
        self.test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        if self.has_labels:
            self.metric = SegmentationMetric(args.num_classes)

        self.model = get_fast_scnn(args.dataset, aux=False, pretrained=False)
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

        # TorchScript JIT trace优化
        if args.use_jit:
            print('Tracing model with TorchScript JIT...')
            example_input = torch.randn(1, 3, args.input_size, args.input_size).to(args.device)
            try:
                traced_model = torch.jit.trace(self.model, example_input)
                # 尝试优化（PyTorch 1.9+支持）
                try:
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                    print('JIT trace + optimize_for_inference 成功！')
                except Exception as e:
                    print(f'JIT trace成功，但optimize_for_inference不可用: {e}')
                self.model = traced_model
            except Exception as e:
                print(f'JIT trace失败，将使用原生模型: {e}')

        # Warmup：让CPU频率稳定，减少首次推理的冷启动影响
        if len(test_dataset) > 0:
            warmup_iters = min(args.warmup_iters, len(test_dataset))
            dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(args.device)
            print(f'Warming up with {warmup_iters} iterations...')
            with torch.no_grad():
                for _ in range(warmup_iters):
                    _ = self.model(dummy_input)

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

                if isinstance(orig_size, torch.Tensor):
                    orig_size = (int(orig_size[0].item()), int(orig_size[1].item()))
                elif isinstance(orig_size, (list, tuple)):
                    if isinstance(orig_size[0], torch.Tensor):
                        orig_size = (int(orig_size[0].item()), int(orig_size[1].item()))
                    else:
                        orig_size = (int(orig_size[0]), int(orig_size[1]))

                batch_start = time.time()
                outputs = self.model(image)
                batch_time = time.time() - batch_start
                inference_times.append(batch_time)

                pred = torch.argmax(outputs[0], 1)
                pred_np = pred.cpu().data.numpy()
                label_np = label.cpu().numpy()

                self.metric.reset()
                self.metric.update(pred_np, label_np)
                metrics = self.metric.get_full_metrics()

                inference_time_ms = batch_time * 1000
                per_image_results.append({
                    'filename': filename,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'miou': metrics['mIoU'],
                    'acc': metrics['pixAcc'],
                    'inference_time_ms': inference_time_ms,
                    'fps': 1000.0 / inference_time_ms if inference_time_ms > 0 else 0.0
                })

                if self.output_dir:
                    self._save_overlay(pred_np.squeeze(0), img_path, orig_size, filename)
                total_images += 1

        test_time = time.time() - test_start
        avg_inference_time_ms = np.mean(inference_times) * 1000 if inference_times else 0
        avg_fps = 1000.0 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0.0
        end_to_end_fps = total_images / test_time if test_time > 0 else 0.0

        overall_metrics = {
            'total_images': total_images,
            'precision': np.mean([r['precision'] for r in per_image_results]),
            'recall': np.mean([r['recall'] for r in per_image_results]),
            'f1': np.mean([r['f1'] for r in per_image_results]),
            'miou': np.mean([r['miou'] for r in per_image_results]),
            'acc': np.mean([r['acc'] for r in per_image_results]),
            'inference_time_ms': avg_inference_time_ms,
            'fps': avg_fps,
            'end_to_end_fps': end_to_end_fps
        }

        print('\n' + '=' * 50)
        print('Raspberry Pi 3B Optimized Evaluation Summary')
        print('=' * 50)
        print(f'Total Images: {total_images}')
        print(f'Input Size: {self.args.input_size}x{self.args.input_size}')
        print(f'JIT Enabled: {self.args.use_jit}')
        print(f'Average Precision: {overall_metrics["precision"]:.6f}')
        print(f'Average Recall:    {overall_metrics["recall"]:.6f}')
        print(f'Average F1-Score:  {overall_metrics["f1"]:.6f}')
        print(f'Average mIoU:      {overall_metrics["miou"]:.6f}')
        print(f'Average Accuracy:  {overall_metrics["acc"]:.6f}')
        print(f'Inference Time:    {avg_inference_time_ms:.4f} ms')
        print(f'Pure Model FPS:    {avg_fps:.2f}')
        print(f'End-to-End FPS:    {end_to_end_fps:.2f}')
        print('=' * 50)

        return per_image_results, overall_metrics

    def _eval_without_labels(self):
        inference_times = []
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

                start = time.time()
                outputs = self.model(image)
                inference_times.append(time.time() - start)

                pred = torch.argmax(outputs[0], 1)
                pred_np = pred.cpu().data.numpy().squeeze(0)

                if self.output_dir:
                    self._save_overlay(pred_np, img_path, orig_size, filename)

        avg_ms = np.mean(inference_times) * 1000 if inference_times else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f'\n无标签推理完成 | 平均推理时间: {avg_ms:.2f} ms | FPS: {fps:.2f}')

    def _save_overlay(self, pred, img_path, orig_size, filename):
        if not self.output_dir or self.args.no_save_overlay:
            return
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        base_name = Path(filename).stem + '.png'

        orig_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = orig_size

        if not self.args.no_overlay_resize:
            if orig_img.size != (orig_w, orig_h):
                orig_img = orig_img.resize((orig_w, orig_h), Image.BILINEAR)
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred_resized = pred_img.resize((orig_w, orig_h), Image.NEAREST)
            pred_mask = (np.array(pred_resized) > 0).astype(np.uint8)
            target_h, target_w = orig_h, orig_w
        else:
            pred_mask = (pred > 0).astype(np.uint8)
            target_h, target_w = pred_mask.shape[0], pred_mask.shape[1]
            if orig_img.size != (target_w, target_h):
                orig_img = orig_img.resize((target_w, target_h), Image.BILINEAR)

        # numpy 向量化 alpha blend，避免 PIL alpha_composite 的额外内存分配
        orig_np = np.array(orig_img, dtype=np.uint16)
        overlay_np = orig_np.copy()
        red_color = np.array([255, 0, 0], dtype=np.uint16)
        mask_3ch = pred_mask[:, :, None]
        overlay_np = (overlay_np * 0.5 + red_color * 0.5).astype(np.uint8)
        result_np = np.where(mask_3ch, overlay_np, orig_np.astype(np.uint8))

        output_path = self.overlay_dir / base_name
        Image.fromarray(result_np).save(output_path)


if __name__ == '__main__':
    args = parse_args()
    evaluator = RPiEvaluator(args)
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
