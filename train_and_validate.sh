#!/bin/bash
export REGION=us-east1
export BUCKET_NAME=gs://amir-bose-asl
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l $REGION $BUCKET_NAME

TRAIN_JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)

gcloud --verbosity=debug ml-engine jobs submit training $TRAIN_JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=$REGION \
--config=youtube-8m/coludml-gpu-distributed_custom.yaml \
-- \
--train_data_pattern='gs://isaacoutputfinal/train*' \
--model=MoeModel \
--train_dir=$BUCKET_NAME/$TRAIN_JOB_NAME \
--feature_names="mean_rgb" \
--feature_sizes="1024" \
--start_new_model = True


VAL_JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)

gcloud --verbosity=debug ml-engine jobs submit training $VAL_JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=$REGION \
--config=youtube-8m/cloudml-gpu.yaml \
-- \
--eval_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/validate/validate*.tfrecord' \
--model=MoeModel \
--train_dir=$BUCKET_NAME/$TRAIN_JOB_NAME \
--feature_names="mean_rgb" \
--feature_sizes="1024" \
--run_once=False