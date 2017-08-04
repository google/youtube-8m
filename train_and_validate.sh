#!/bin/bash
if [ $# -eq 0 ]; then
    BUCKET_NAME=gs://amir-bose-asl
else
    BUCKET_NAME=$1
fi

REGION=us-east1
MODEL="MoeModel"

MOE_NUM_MIXTURES=2

FEATURE_NAMES="mean_rgb,mean_audio"
FEATURE_SIZES="1024,128"
BATCH_SIZE=256

# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l $REGION $BUCKET_NAME

TRAIN_JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S)

gcloud --verbosity=debug ml-engine jobs submit training $TRAIN_JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=$REGION \
--config=youtube-8m/train.yaml \
-- \
--train_data_pattern="gs://isaacoutputfinal/train*" \
--model=$MODEL \
--moe_num_mixtures=$MOE_NUM_MIXTURES \
--train_dir=$BUCKET_NAME/$TRAIN_JOB_NAME \
--feature_names=$FEATURE_NAMES \
--feature_sizes=$FEATURE_SIZES \
--batch_size=$BATCH_SIZE \
--num_epochs=10 \
--moe_num_mixtures=8 \
--start_new_model = True


VAL_JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S)

gcloud --verbosity=debug ml-engine jobs submit training $VAL_JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=$REGION \
--config=youtube-8m/validate.yaml \
-- \
--eval_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/validate/validate*.tfrecord' \
--model=$MODEL \
--moe_num_mixtures=$MOE_NUM_MIXTURES \
--train_dir=$BUCKET_NAME/$TRAIN_JOB_NAME \
--feature_names=$FEATURE_NAMES \
--feature_sizes=$FEATURE_SIZES \
--batch_size=$BATCH_SIZE \
--run_once=False
