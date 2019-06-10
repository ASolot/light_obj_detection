#!/bin/bash
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet
# train
# python train.py ctdet --exp_id visdrone_dla_1x --batch_size 32 --dataset visdrone --lr 5e-4 --gpus 0 --num_workers 4
python train.py ctdet --exp_id visdrone_dla_1x --arch res_18 --batch_size 32 --dataset visdrone --lr 5e-4 --gpus 0 --num_workers 4 --num_epochs 5
# test
python evaluate.py ctdet --exp_id visdrone_dla_1x --arch res_18 --dataset visdrone --keep_res --resume
# flip test
python evaluate.py ctdet --exp_id visdrone_dla_1x --arch res_18 --dataset visdrone --keep_res --resume --flip_test 
# multi scale test
python evaluate.py ctdet --exp_id visdrone_dla_1x --arch res_18 --dataset visdrone --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
