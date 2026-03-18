import os
import argparse
import time
import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
# from data_loader import get_segmentation_dataset
from data_loader.water_val import CustomSegmentationDataset  # 导入自定义数据集
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
    parser.add_argument('--dataset', type=str, default='custom',
                        help='dataset name (default: custom)')
    
    # ========== 新增：直接指定路径的参数 ==========
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
    # ============================================
    
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
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,  # 修改为默认False，启用验证
                        help='skip validation during training')
    # the parser
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    
    # 检查路径参数
    if args.images is None or args.masks is None:
        parser.error("--images 和 --masks 参数必须指定")
    if not args.no_val and (args.val_images is None or args.val_masks is None):
        parser.error("启用验证时，--val-images 和 --val-masks 参数必须指定")
    
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # ========== 修改：使用自定义数据集 ==========
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
        # 设置类别数
        
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
        # ===========================================
        
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=4,  # 添加多进程加载
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

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight,
                                                        ignore_index=-1).to(args.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        cur_iters = 0
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
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

                epoch_loss += loss.item()
                cur_iters += 1
                
                if cur_iters % 10 == 0:
                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(self.train_loader),
                        time.time() - start_time, cur_lr, loss.item()))

            avg_loss = epoch_loss / len(self.train_loader)
            print(f'Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss:.4f}')

            if self.args.no_val:
                # save every epoch
                save_checkpoint(self.model, self.args, is_best=False)
            else:
                self.validation(epoch)

        save_checkpoint(self.model, self.args, is_best=False)

    def validation(self, epoch):
        is_best = False
        self.metric.reset()
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i, (image, target) in enumerate(self.val_loader):
                image = image.to(self.args.device)
                target = target.to(self.args.device)

                outputs = self.model(image)
                loss = self.criterion(outputs, target)
                val_loss += loss.item()
                
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                self.metric.update(pred, target.cpu().numpy())
                
        pixAcc, mIoU = self.metric.get()
        avg_val_loss = val_loss / len(self.val_loader)
        
        print('Epoch %d, validation pixAcc: %.3f%%, mIoU: %.3f%%, Loss: %.4f' % (
            epoch, pixAcc * 100, mIoU * 100, avg_val_loss))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            print(f'>>> 新的最佳模型! 得分: {new_pred:.4f}')
        save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_epoch{}.pth'.format(args.model, args.dataset, args.epochs)
    save_path = os.path.join(directory, filename)
    
    # 保存模型状态
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, save_path)
    print(f'模型已保存: {save_path}')
    
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_path = os.path.join(directory, best_filename)
        shutil.copyfile(save_path, best_path)
        print(f'最佳模型已保存: {best_path}')


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluation model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
        print(f'训练图像: {args.images}')
        print(f'训练掩码: {args.masks}')
        if not args.no_val:
            print(f'验证图像: {args.val_images}')
            print(f'验证掩码: {args.val_masks}')
        trainer.train()