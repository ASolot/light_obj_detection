#!/bin/bash
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet
# train
python train.py ctdet --exp_id visdrone_dla_512 --batch_size 16 --dataset visdrone --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 0 --resume
# test
python evaluate.py ctdet --exp_id visdrone_dla_512 --dataset visdrone --input_res 512 --resume
# flip test
python evaluate.py ctdet --exp_id visdrone_dla_512 --dataset visdrone --input_res 512 --resume --flip_test
