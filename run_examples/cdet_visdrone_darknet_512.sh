#!/bin/bash
cd $HOME/MThesis/repos/mine/light_obj_detection/SolotNet
# train -> TODO: change learning rate and the number of epochs and lr step
python train.py ctdet --exp_id visdrone_darknet_512 --arch darknet --dataset visdrone --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 0
# test
python evaluate.py ctdet --exp_id visdrone_darknet_512 --arch darknet --dataset visdrone --input_res 512 --resume
# flip test
python evaluate.py ctdet --exp_id visdrone_darknet_512 --arch darknet --dataset visdrone --input_res 512 --resume --flip_test
