# 模型推理（树莓派3B）
python eval_water_pro.py \
    --input "/home/pi/Data/images" \
    --mask "/home/pi/Data/masks_Ids" \
    --model-path "/home/pi/Data/experiment1_last.pth" \
    --output ./test_results_pro/


python eval_water_pro.py \
    --input "/home/pi/Data/Test/images" \
    --mask "/home/pi/Data/Test/masks_Ids" \
    --model-path "/home/pi/Data/experiment1_last.pth" \
    --output ./test_results_test/


python eval_water_pro.py \
    --input "/home/pi/Data/WelaBoat/images" \
    --mask "/home/pi/Data/WelaBoat/masks_Ids" \
    --model-path "/home/pi/Data/experiment1_last.pth" \
    --output ./test_results_WelaBoat/


# 模型推理（树莓派3B）优化加速版
# 推荐测试组合 A（平衡精度与速度，默认）
python eval_water_rpi.py \
    --input "/home/pi/Data/images" \
    --mask "/home/pi/Data/masks_Ids" \
    --model-path "/home/pi/Data/experiment1_last.pth" \
    --input-size 256 \
    --use-jit \
    --output ./test_results_pro_rpi/

#推荐测试组合 B（极限速度，跳过图片保存）
python eval_water_rpi.py \
    --input "/home/pi/Data/images" \
    --mask "/home/pi/Data/masks_Ids" \
    --model-path "/home/pi/Data/experiment1_last.pth" \
    --input-size 256 \
    --use-jit \
    --no-save-overlay \
    --output ./test_results_pro_rpi2/

#推荐测试组合 C (降到 192，牺牲一点精度换速度)
python eval_water_rpi.py \
    --input "/home/pi/Data/images" \
    --mask "/home/pi/Data/masks_Ids" \
    --model-path "/home/pi/Data/experiment1_last.pth" \
    --input-size 192 \
    --use-jit \
    --no-save-overlay \
    --output ./test_results_pro_rpi3/