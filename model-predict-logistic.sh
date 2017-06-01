#!/bin/bash
MODEL=logistic
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--output_file=/models/$MODEL/predictions.csv \
	--input_data_pattern='/data/video/video-level-features/test.tfrecord' \
	--train_dir=/models/$MODEL
