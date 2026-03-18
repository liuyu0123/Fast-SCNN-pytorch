from __future__ import division

import threading
import numpy as np

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']


class SegmentationMetric(object):
    """Computes pixAcc, mIoU, Precision, Recall, and F1 metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        if isinstance(preds, np.ndarray):
            self.evaluate_worker(preds, labels)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=self.evaluate_worker, args=(pred, label), )
                       for (pred, label) in zip(preds, labels)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        # It has same result with np.nanmean() when all class exist
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def get_full_metrics(self):
        """Gets comprehensive metrics including Precision, Recall, F1
        
        Returns
        -------
        metrics : dict
            包含 pixAcc, mIoU, precision, recall, f1
        """
        # 从混淆矩阵计算各项指标
        hist = self.confusion_matrix
        
        # 计算每个类别的 precision, recall, f1
        # Precision = TP / (TP + FP) = 对角线 / 列和
        # Recall = TP / (TP + FN) = 对角线 / 行和
        
        # 防止除零
        eps = np.spacing(1)
        
        # 计算每个类别的指标
        tp = np.diag(hist)  # True Positives (对角线)
        
        # Precision per class: TP / (TP + FP) = diag / col_sum
        col_sum = hist.sum(axis=0)
        precision_per_class = tp / (col_sum + eps)
        
        # Recall per class: TP / (TP + FN) = diag / row_sum  
        row_sum = hist.sum(axis=1)
        recall_per_class = tp / (row_sum + eps)
        
        # F1 per class
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)
        
        # IoU per class (和原来的 mIoU 计算一致)
        union = row_sum + col_sum - tp
        iou_per_class = tp / (union + eps)
        
        # 计算宏平均 (macro-average) - 对所有类别平均，包括背景
        # 通常语义分割中，背景类(0)也参与计算，但如果想排除背景，可以用 [1:]
        precision_macro = np.nanmean(precision_per_class)
        recall_macro = np.nanmean(recall_per_class)
        f1_macro = np.nanmean(f1_per_class)
        mIoU = np.nanmean(iou_per_class)
        
        # 计算微平均 (micro-average) - 基于总体 TP/FP/FN
        tp_total = tp.sum()
        precision_micro = tp_total / (col_sum.sum() + eps)
        recall_micro = tp_total / (row_sum.sum() + eps)
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)
        
        # 像素准确率
        pixAcc = 1.0 * self.total_correct / (self.total_label + eps)
        
        return {
            'pixAcc': pixAcc,
            'mIoU': mIoU,
            'precision': precision_macro,      # 宏平均精度
            'recall': recall_macro,            # 宏平均召回
            'f1': f1_macro,                    # 宏平均 F1
            'precision_micro': precision_micro,  # 微平均精度
            'recall_micro': recall_micro,        # 微平均召回
            'f1_micro': f1_micro,                # 微平均 F1
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'iou_per_class': iou_per_class.tolist(),
            'confusion_matrix': hist.tolist()
        }

    def evaluate_worker(self, pred, label):
        # 更新混淆矩阵
        hist, labeled, correct = hist_info(pred, label, self.nclass)
        inter, union = batch_intersection_union(pred, label, self.nclass)
        
        with self.lock:
            self.confusion_matrix += hist
            self.total_correct += correct
            self.total_label += labeled
            self.total_inter += inter
            self.total_union += union

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        # 新增：混淆矩阵，用于计算 precision/recall/f1
        self.confusion_matrix = np.zeros((self.nclass, self.nclass), dtype=np.int64)


def batch_pix_accuracy(predict, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    assert predict.shape == target.shape
    predict = predict.astype('int64') + 1
    target = target.astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    assert predict.shape == target.shape
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.astype('int64') + 1
    target = target.astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter, area_union


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    """
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    """
    imPred = imPred * (imLab >= 0)

    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    """计算混淆矩阵"""
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    # 构建混淆矩阵: hist[i, j] 表示真实为i预测为j的像素数
    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc