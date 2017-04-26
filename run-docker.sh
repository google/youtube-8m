# set defaults
USE_GPU=false
HOST_PORT=9999
GPU_IMAGE=tensorflow/tensorflow:latest-gpu
CPU_IMAGE=mzak/pipeline

DATA=/mnt/data/
MODELS=/mnt/models/video
NAME=video_tags

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
    echo "Running docker on GPU"
    nvidia-docker run -it -d \
        --volume=$MODELS:/models \
        --volume=$DATA:/data \
	--name $NAME \
	--volume=$(pwd):/workspace $GPU_IMAGE /bin/bash
else
    echo "Running docker on CPU"
    docker run -it -d \
	    --volume=$MODELS:/models \
	    --volume=$DATA:/data \
	    --name $NAME \
	    --volume=$(pwd):/workspace $CPU_IMAGE /bin/bash
fi
