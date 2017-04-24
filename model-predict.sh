#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
	--output_file=/models/Lstm/predictions.csv \
	--input_data_pattern='/data/frame-level/test/test*.tfrecord' \
	--train_dir=/models/Lstm \
	--frame_features=True \
	--feature_names="rgb" \
	--feature_sizes="1024"

