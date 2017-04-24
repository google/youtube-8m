#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python eval.py  \
	--eval_data_pattern='/data/frame-level/validate/validate*.tfrecord' \
	--model=LstmModel \
	--train_dir=/models/Lstm/ \
	--run_once=True \
	--frame_features=True \
	--feature_names="rgb" \
	--feature_sizes="1024"
