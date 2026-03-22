# 训练
python train.py --model fast_scnn --dataset citys
# 训练自己的数据集
python train.py --model fast_scnn --dataset water
# 训练模型（水域分割，train和val分离）
python train_water.py `
    --model fast_scnn `
    --dataset water `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_Ids `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_Ids `
    --num-classes 2 `
    --batch-size 4 `
    --epochs 5 `
    --lr 0.01

# 训练模型(pro)
python train_water_val_pro.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_Ids `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_Ids `
    --epochs 5 `
    --batch-size 4 `
    --learning-rate 5e-4 `
    --model-dir checkpoints/experiment1 `
    --log-dir logs/experiment1 `
    --model-name experiment1 `
    --log-name experiment1 `
    --save-interval 0

# 测试
python eval.py
# 测试自己的数据集(需要修改下models\fast_scnn.py)，训练完成后生成：weights\fast_scnn_water.pth
python eval.py --model fast_scnn --dataset water
# 测试（水域分割，指定test路径并记录预测结果csv）
#1. 带标签测试（计算指标）✅
python eval_water.py `
    --model-path D:\Files\GitProject\Fast-SCNN-pytorch-LY\weights\fast_scnn_water_best_model.pth `
    --test-images D:\Files\Data\IRWSB\test\images `
    --test-masks D:\Files\Data\IRWSB\test\masks_Ids `
    --dataset water `
    --num-classes 2 `
    --save-mask `
    --save-overlay `
    --output weights/test_result.csv
#2. 无标签测试（仅推理）
python eval_water.py `
    --model-path D:\Files\GitProject\Fast-SCNN-pytorch-LY\weights\fast_scnn_water_best_model.pth `
    --test-images D:\Files\Data\IRWSB\test\images `
    --dataset water `
    --num-classes 2 `
    --save-mask

# 演示，处理单张图片
python demo.py --model fast_scnn --input-pic './png/berlin_000000_000019_leftImg8bit.png'
python demo.py --model fast_scnn --input-pic "D:\Files\Data\StereoCamera\ImageStereo\RiverImage\past\sync_20251227_121116_495.png"
# 演示，处理单张图片，河道
python demo.py --model fast_scnn --dataset water --input-pic "D:\Files\Data\StereoCamera\ImageStereo\RiverImage\past\sync_20251227_121116_495.png"
python demo.py --model fast_scnn --dataset water --input-pic "D:\Files\Data\StereoCamera\ImageStereo\Data2_HT_notgood\image2\left\008.png"
# 演示，处理无人船实验图片USVInlandDataset数据集
python demo.py --model fast_scnn --dataset water --input-pic "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\1280_640_undistorted\X01_1_0000010900.jpg"
# 演示，处理无人船实验图片
python demo.py --model fast_scnn --dataset water --input-pic "D:\Files\Data\StereoUSV\dataset1\left\stereo_20260108_110215\000002.png"
# 演示，处理视频（无法运行）
python demo.py --model fast_scnn --input-pic "D:\Files\Data\StereoCamera\ImageStereo\RiverVideo\past\stereo_vfr_20251227_121002.mp4"
