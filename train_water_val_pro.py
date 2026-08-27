import os
import csv
import argparse
import time
import shutil
from pathlib import Path

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np

from torchvision import transforms

from data_loader.water_val import CustomSegmentationDataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='water',
                        help='dataset name (default: water)')
    
    # 路径参数
    parser.add_argument('--images', type=str, default=None,
                        help='训练图像文件夹路径')
    parser.add_argument('--masks', type=str, default=None,
                        help='训练掩码文件夹路径')
    parser.add_argument('--val-images', type=str, default=None,
                        help='验证图像文件夹路径')
    parser.add_argument('--val-masks', type=str, default=None,
                        help='验证掩码文件夹路径')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='分割类别数（包括背景）')
    
    # 新增：保存路径和文件名控制参数
    parser.add_argument('--save-folder', default='./weights',
                        help='默认保存目录（如果未指定--model-dir和--log-dir）')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='模型保存目录（默认使用 --save-folder）')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='日志保存目录（默认使用 --save-folder）')
    parser.add_argument('--model-name', type=str, default=None,
                        help='模型保存文件名前缀（默认：{model}_{dataset}）')
    parser.add_argument('--log-name', type=str, default=None,
                        help='日志文件名前缀（默认：{model}_{dataset}_training_log）')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='分步保存频率，每N个epoch保存一次中间模型，0为不保存（默认：0）')
    
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for training (default: 12)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    
    # 检查路径参数
    if args.images is None or args.masks is None:
        parser.error("--images 和 --masks 参数必须指定")
    if not args.no_val and (args.val_images is None or args.val_masks is None):
        parser.error("启用验证时，--val-images 和 --val-masks 参数必须指定")
    
    # 设置默认路径：如果未指定model-dir/log-dir，使用save-folder
    if args.model_dir is None:
        args.model_dir = args.save_folder
    if args.log_dir is None:
        args.log_dir = args.save_folder
    
    # 设置默认文件名：如果未指定model-name/log-name，使用默认值
    if args.model_name is None:
        args.model_name = f"{args.model}_{args.dataset}"
    if args.log_name is None:
        args.log_name = f"{args.model}_{args.dataset}_training_log"
    
    # 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
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


class MetricsLogger:
    """指标记录器，支持实时写入CSV（每个epoch写入一次）"""
    
    def __init__(self, save_path, model_info, args):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        self.args = args
        
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        
        # 立即创建文件并写入表头（覆盖模式）
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
        print(f"Log file initialized: {self.save_path}")
        
    def log_epoch(self, epoch, train_loss, train_metrics, val_loss, val_metrics, 
                  inference_time_ms, fps, lr):
        """记录一轮数据并立即写入文件（追加模式）"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'train_precision': f"{train_metrics.get('precision', 0):.6f}",
            'train_recall': f"{train_metrics.get('recall', 0):.6f}",
            'train_f1': f"{train_metrics.get('f1', 0):.6f}",
            'train_miou': f"{train_metrics.get('miou', 0):.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_precision': f"{val_metrics.get('precision', 0):.6f}",
            'val_recall': f"{val_metrics.get('recall', 0):.6f}",
            'val_f1': f"{val_metrics.get('f1', 0):.6f}",
            'val_miou': f"{val_metrics.get('miou', 0):.6f}",
            'inference_time_ms': f"{inference_time_ms:.4f}",
            'fps': f"{fps:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        
        # 立即追加写入CSV，不缓存
        with open(self.save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(row)
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.save_path.parent / f"{self.save_path.stem}_model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model Type: {self.args.model}\n")
            f.write(f"Dataset: {self.args.dataset}\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Input Size: {self.args.base_size}x{self.args.crop_size}\n")
            f.write(f"Num Classes: {self.args.num_classes}\n")
            f.write(f"Epochs: {self.args.epochs}\n")
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Learning Rate: {self.args.lr}\n")
            f.write(f"Model Name Prefix: {self.args.model_name}\n")
            f.write(f"Save Interval: {self.args.save_interval}\n")


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # 训练数据集
        train_dataset = CustomSegmentationDataset(
            images_dir=args.images,
            masks_dir=args.masks,
            transform=input_transform,
            base_size=args.base_size,
            crop_size=args.crop_size,
            mode='train',
            num_classes=args.num_classes
        )
        
        # 验证数据集
        if not args.no_val:
            val_dataset = CustomSegmentationDataset(
                images_dir=args.val_images,
                masks_dir=args.val_masks,
                transform=input_transform,
                base_size=args.base_size,
                crop_size=args.crop_size,
                mode='val'
            )
            val_dataset.num_class = args.num_classes
        
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=4,
                                            pin_memory=True)
        if not args.no_val:
            self.val_loader = data.DataLoader(dataset=val_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)
        else:
            self.val_loader = None

        # create network
        self.model = get_fast_scnn(dataset=args.dataset, aux=args.aux)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to(args.device)

        # 获取模型信息并创建记录器
        self.model_info = get_model_info(self.model)
        print(f"Model: {self.model_info['total_params']:,} params, {self.model_info['model_size_mb']:.2f} MB")
        
        # 创建记录器 - 使用指定的日志路径和文件名
        log_filename = f"{args.log_name}.csv"
        self.metrics_logger = MetricsLogger(
            os.path.join(args.log_dir, log_filename),
            self.model_info,
            args
        )
        self.metrics_logger.save_model_info()

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(
            aux=args.aux, 
            aux_weight=args.aux_weight,
            ignore_index=-1
        ).to(args.device)

        # optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        # lr scheduling
        self.lr_scheduler = LRScheduler(
            mode='poly', 
            base_lr=args.lr, 
            nepochs=args.epochs,
            iters_per_epoch=len(self.train_loader), 
            power=0.9
        )

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        
        # 训练集指标计算器
        self.train_metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0
        self.best_miou = 0.0

    def train(self):
        cur_iters = 0
        start_time = time.time()
        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            self.train_metric.reset()
            epoch_loss = 0.0

            for i, (images, targets) in enumerate(self.train_loader):
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 计算训练指标
                with torch.no_grad():
                    pred = torch.argmax(outputs[0] if isinstance(outputs, tuple) else outputs, 1)
                    pred = pred.cpu().data.numpy()
                    self.train_metric.update(pred, targets.cpu().numpy())

                epoch_loss += loss.item()
                cur_iters += 1
                
                if cur_iters % 10 == 0:
                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                        epoch, self.args.epochs, i + 1, len(self.train_loader),
                        time.time() - start_time, cur_lr, loss.item()))

            # 计算训练指标
            train_metrics_full = self.train_metric.get_full_metrics()
            train_metrics = {
                'precision': train_metrics_full['precision'],
                'recall': train_metrics_full['recall'],
                'f1': train_metrics_full['f1'],
                'miou': train_metrics_full['mIoU'],
            }
            avg_train_loss = epoch_loss / len(self.train_loader)
            
            print(f'Epoch [{epoch}/{self.args.epochs}] Average Loss: {avg_train_loss:.4f}, Train mIoU: {train_metrics["miou"]:.4f}')

            if self.args.no_val:
                # 无验证模式：记录训练指标，验证指标留空
                self.metrics_logger.log_epoch(
                    epoch + 1,
                    avg_train_loss,
                    train_metrics,
                    0,  # val_loss
                    {'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0},  # val_metrics
                    0,  # inference_time_ms
                    0,  # fps
                    cur_lr
                )
                
                # 分步保存中间模型（仅在无验证模式下，因为验证模式在validation函数中处理）
                is_interval_save = (self.args.save_interval > 0) and ((epoch + 1) % self.args.save_interval == 0)
                if is_interval_save:
                    save_checkpoint(self.model, self.args, epoch=epoch+1, is_interval=True)
                    
            else:
                # 验证并记录
                self.validation(epoch, avg_train_loss, train_metrics, cur_lr)

        # 训练结束，保存最终模型（last）
        save_checkpoint(self.model, self.args, is_last=True)
        print(f"\nTraining complete! Final model saved as {self.args.model_name}_last.pth")
        print(f"Best Val mIoU: {self.best_miou:.4f}")

    def validation(self, epoch, train_loss, train_metrics, train_lr):
        is_best = False
        self.metric.reset()
        self.model.eval()
        val_loss = 0.0
        
        # 测量推理时间
        inference_times = []
        total_val_images = 0
        
        val_start = time.time()
        
        with torch.no_grad():
            for i, (image, target) in enumerate(self.val_loader):
                image = image.to(self.args.device)
                target = target.to(self.args.device)

                # 测量推理时间
                if self.args.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_start = time.time()
                
                outputs = self.model(image)
                
                if self.args.device.type == 'cuda':
                    torch.cuda.synchronize()
                batch_time = time.time() - batch_start
                inference_times.append(batch_time)
                
                loss = self.criterion(outputs, target)
                val_loss += loss.item()
                
                pred = torch.argmax(outputs[0] if isinstance(outputs, tuple) else outputs, 1)
                pred = pred.cpu().data.numpy()
                self.metric.update(pred, target.cpu().numpy())
                total_val_images += image.size(0)
                
        val_time = time.time() - val_start
        
        val_metrics_full = self.metric.get_full_metrics()
        avg_val_loss = val_loss / len(self.val_loader)
        pixAcc = val_metrics_full['pixAcc']
        mIoU = val_metrics_full['mIoU']
        
        # 计算验证指标
        val_metrics = {
            'precision': val_metrics_full['precision'],
            'recall': val_metrics_full['recall'],
            'f1': val_metrics_full['f1'],
            'miou': val_metrics_full['mIoU'],
        }
        
        # 计算推理时间
        avg_inference_time_ms = np.mean(inference_times) * 1000
        fps = total_val_images / val_time if val_time > 0 else 0
        
        print('Epoch %d, validation pixAcc: %.3f%%, mIoU: %.3f%%, Precision: %.3f%%, Recall: %.3f%%, F1: %.3f%%, Loss: %.4f, FPS: %.2f' % (
            epoch, pixAcc * 100, mIoU * 100, 
            val_metrics['precision'] * 100, val_metrics['recall'] * 100, val_metrics['f1'] * 100,
            avg_val_loss, fps))

        # 记录到CSV（实时写入）
        self.metrics_logger.log_epoch(
            epoch + 1,
            train_loss,
            train_metrics,
            avg_val_loss,
            val_metrics,
            avg_inference_time_ms,
            fps,
            train_lr
        )

        # 检查是否为最佳模型
        new_pred = (pixAcc + mIoU) / 2
        if mIoU > self.best_miou:
            is_best = True
            self.best_miou = mIoU
            self.best_pred = new_pred
            print(f'>>> 新的最佳模型! mIoU: {mIoU:.4f}, 得分: {new_pred:.4f}')
            # 保存最佳模型
            save_checkpoint(self.model, self.args, is_best=True)
        
        # 分步保存中间模型（按interval）
        is_interval_save = (self.args.save_interval > 0) and ((epoch + 1) % self.args.save_interval == 0)
        if is_interval_save:
            save_checkpoint(self.model, self.args, epoch=epoch+1, is_interval=True)


def save_checkpoint(model, args, epoch=None, is_best=False, is_last=False, is_interval=False):
    """
    保存检查点
    命名规则：
    - 最佳模型: {model_name}_best.pth
    - 最终模型: {model_name}_last.pth  
    - 中间模型: {model_name}_epoch{epoch}.pth（仅在save_interval>0时生成）
    """
    # 确定文件名
    if is_best:
        filename = f"{args.model_name}_best.pth"
        print(f'>>> 保存最佳模型: {filename}')
    elif is_last:
        filename = f"{args.model_name}_last.pth"
        print(f'保存最终模型: {filename}')
    elif is_interval and epoch is not None:
        filename = f"{args.model_name}_epoch{epoch}.pth"
        print(f'保存中间模型 (epoch {epoch}): {filename}')
    else:
        # 默认情况（兼容旧代码，不应在优化后的逻辑中调用）
        filename = f"{args.model_name}_epoch{epoch if epoch else args.epochs}.pth"
    
    save_path = os.path.join(args.model_dir, filename)
    
    # 保存模型状态
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, save_path)
    
    # 如果是最佳模型，额外复制一份（可选，保留最新best）
    if is_best:
        best_path = os.path.join(args.model_dir, f"{args.model_name}_best.pth")
        # 已经以此名保存，无需复制
        pass


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluation model: ', args.resume)
        trainer.validation(args.start_epoch, 0, {'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0}, args.lr)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
        print(f'训练图像: {args.images}')
        print(f'训练掩码: {args.masks}')
        print(f'模型保存目录: {args.model_dir}')
        print(f'日志保存目录: {args.log_dir}')
        print(f'模型名称前缀: {args.model_name}')
        print(f'日志名称前缀: {args.log_name}')
        print(f'分步保存间隔: {args.save_interval} (0为不保存中间模型)')
        if not args.no_val:
            print(f'验证图像: {args.val_images}')
            print(f'验证掩码: {args.val_masks}')
        trainer.train()