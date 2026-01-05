# 训练
python train.py --model fast_scnn --dataset citys
# 训练自己的数据集
python train.py --model fast_scnn --dataset water

# 测试
python eval.py
# 测试自己的数据集(需要修改下models\fast_scnn.py)
python eval.py --model fast_scnn --dataset water

# 演示，处理单张图片
python demo.py --model fast_scnn --input-pic './png/berlin_000000_000019_leftImg8bit.png'
python demo.py --model fast_scnn --input-pic "D:\Files\Data\StereoCamera\ImageStereo\RiverImage\past\sync_20251227_121116_495.png"
# 演示，处理单张图片，河道
python demo.py --model fast_scnn --dataset water --input-pic "D:\Files\Data\StereoCamera\ImageStereo\RiverImage\past\sync_20251227_121116_495.png"
python demo.py --model fast_scnn --dataset water --input-pic "D:\Files\Data\StereoCamera\ImageStereo\Data2_HT_notgood\image2\left\008.png"
# 演示，处理视频（无法运行）
python demo.py --model fast_scnn --input-pic "D:\Files\Data\StereoCamera\ImageStereo\RiverVideo\past\stereo_vfr_20251227_121002.mp4"
