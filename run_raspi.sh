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