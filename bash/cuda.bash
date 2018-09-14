# This code is intended to be used in all test cases which require Cuda. This
# are most likely the test cases run on Jenkins which call TensorFlow. This file
# can be sourced in your own .bashrc or .env file.

# CuDNN
export CUDNN=/net/ssd/software/cudnn/v7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDNN}/lib64

if [[ $(hostname) =~ ^ntsim.*|ntpc9|ntjenkins ]];
then
    # Cuda libraries for GPU hosts
    export CUDA_HOME=/usr/local/cuda
    export PATH=$PATH:${CUDA_HOME}/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/nvvm/libdevice
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/extras/CUPTI/lib64
    export CUDA_VISIBLE_DEVICES=0
else
    # Cuda libraries for non-GPU hosts to allow Tensorflow to run there, too.
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/ssd/software/nvidia_lib
    export CUDA_VISIBLE_DEVICES=
fi
