#!/bin/bash
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet
# train
python train.py ctdet --exp_id visdrone_res_1024 --batch_size 32  --arch resdcn_18 --dataset visdrone --input_res 1024 --num_epochs 50 --lr 5e-4 --lr_step 45,60 --gpus 0,1,2,3 --num_workers 16
# test
python evaluate.py ctdet --exp_id visdrone_res_1024 --dataset visdrone --arch resdcn_18 --keep_res --resume
# flip test
python evaluate.py ctdet --exp_id visdrone_res_1024 --dataset visdrone --arch resdcn_18 --keep_res --resume --flip_test

