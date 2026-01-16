# 训练
python train.py --model fast_scnn --dataset citys
# 训练自己的数据集
python train.py --model fast_scnn --dataset water

# 测试
python eval.py
# 测试自己的数据集(需要修改下models\fast_scnn.py)，训练完成后生成：weights\fast_scnn_water.pth
python eval.py --model fast_scnn --dataset water

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
