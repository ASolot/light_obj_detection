#!/bin/bash
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet
# train
python train.py ctdet --exp_id visdrone_eesp_512 --arch espnetv2 --dataset visdrone --input_res 512 --num_epochs 70 --gpus 0
# test
python evaluate.py ctdet --exp_id visdrone_eesp_512 --arch espnetv2 --dataset visdrone --input_res 512 --resume
# flip test
python evaluate.py ctdet --exp_id visdrone_eesp_512 --arch espnetv2 --dataset visdrone --input_res 512 --resume --flip_test
