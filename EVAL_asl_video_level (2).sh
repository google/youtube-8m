#!/bin/bash
#Sample command lines to run the cat_vs_dog example in Google Cloud

# This sample assumes you're already setup for using CloudML.  If this is your
# first time with the service, start here:
# https://cloud.google.com/ml/docs/how-tos/getting-set-up



# Declare some environment variables used in data preprocessing and training.

# The following variables may need to be changed based on your own config.

BUCKET_NAME=gs://mrudula_yt8m_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-east1 $BUCKET_NAME

JOB_TO_EVAL=yt8m_train_20170803_015910
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/validate/validate*.tfrecord' \
--model=MoeModel \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --run_once=False