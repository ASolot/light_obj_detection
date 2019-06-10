#!/bin/bash
# script for loading modules and environment 
# configuration on the wilkies2 cluster

#module purge

# conda environment
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
module unload cuda/8.0 
module load cuda/10.0
module load cuda/9.0 intel/mkl/2017.4 
module load cudnn/7.3_cuda-9.0

source activate thesisenvcl
cd "$HOME/MThesis/repos/mine/light_obj_detection/SolotNet"
# tb-nightly future 
