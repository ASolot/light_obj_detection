#!/bin/sh

git clone https://github.com/AlexeyAB/darknet.git

cd darknet

make GPU=1 CUDNN=1 CUDNN_HALF=1 OPENCV=1 OPENMP=1
