#!/bin/bash
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet
# train
python train.py ctdet --exp_id visdrone_dla_1024 --batch_size 16 --dataset visdrone --input_res 1024 --num_epochs 80 --lr_step 45,60 --gpus 0
# test
python evaluate.py ctdet --exp_id visdrone_dla_1024 --dataset visdrone --input_res 1024 --resume
# flip test
python evaluate.py ctdet --exp_id visdrone_dla_1024 --dataset visdrone --input_res 1024 --resume --flip_test
