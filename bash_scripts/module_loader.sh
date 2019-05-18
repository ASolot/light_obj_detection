# script for loading modules and environment 
# configuration on the wilkies2 cluster

module purge

# conda environment
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz

# # CUDA 10.0 & cudnn
# module load cuda/10.0 intel/mkl/2017.4 cudnn/7.5_cuda-10.0

module unload cuda/8.0
# for unknown reasons, tensorboard won't work 
module load cuda/10.0

# CUDA 9.0 & cudnn 
module load cuda/9.0 intel/mkl/2017.4 
module load cudnn/7.3_cuda-9.0

# tb-nightly future 
