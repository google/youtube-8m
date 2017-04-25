# set defaults
USE_GPU=false
HOST_PORT=9999
GPU_IMAGE=tensorflow/tensorflow:latest-gpu
CPU_IMAGE=purbanski/pca_matrix

DATA=/mnt/data/kaggle
MODELS=/mnt/models/video

# parse arguments
while getopts gd:m:p: option
do 
    case "${option}" in 
        g) USE_GPU='true';;
        p) HOST_POST=${OPTARG};; 
        d) DATA=${OPTARG};; 
        m) MODELS=${OPTARG};; 
        *) error "unexpected option ${flag}";;
    esac
done

if $USE_GPU
then
    echo "Running caffe on GPU"
    nvidia-docker run -it -d \
        -p $HOST_PORT:8888 \
        --log-driver=journald \
        --volume=$MODELS:/models \
        --volume=$DATA:/data \
        --volume=$(pwd):/workspace $GPU_IMAGE /bin/bash 
else
    echo "Running caffe on CPU"
    docker run -it -d \
        -p $HOST_PORT:8888  \
        --log-driver=journald \
        --volume=$MODELS:/models \
        --volume=$DATA:/data \
        --volume=$(pwd):/workspace $CPU_IMAGE /bin/bash
fi
