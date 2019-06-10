#!/bin/bash
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet
# train
python train.py ctdet --exp_id visdrone_esp_c_1024 --batch_size 32 --arch espnetv2 --dataset visdrone --input_res 1024 --num_epochs 50 --lr_step 45,60 --gpus 0,1,2,3 --num_workers 32 --resume
# test
python evaluate.py ctdet --exp_id visdrone_esp_c_1024 --arch espnetv2 --dataset visdrone --input_res 1024 --resume
# flip test
python evaluate.py ctdet --exp_id visdrone_esp_c_1024 --arch espnetv2 --dataset visdrone --input_res 1024 --resume --flip_test
