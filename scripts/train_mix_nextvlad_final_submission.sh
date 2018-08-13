#!/usr/bin/env bash

datapath=/media/linrongc/dream/data/yt8m/2/frame
eval_path=$HOME/datasets/competitions/yt8m/2/val
test_path=/media/linrongc/dream/data/yt8m/2/frame/test

model_name=MixNeXtVladModel
parameters="--groups=8 --nextvlad_cluster_size=112 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=16  --drop_rate=0.5 \
            --mix_number=3 --cl_temperature=3 --cl_lambda=9"

train_dir=mix3_nextvlad_3T_8g_5l2_5drop_112k_2048_2x80_logistic
result_folder=results

echo "model name: " $model_name
echo "model parameters: " $parameters

echo "training directory: " $train_dir
echo "data path: " $datapath
echo "evaluation path: " $eval_path
echo "results folder: " $result_folder

python train.py ${parameters} --model=${model_name} --num_readers=8 --learning_rate_decay_examples 2500000 --num_epochs=15\
                --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=False \
                --train_data_pattern=${datapath}/[tv][ar]*/[tv]*.tfrecord --train_dir=${train_dir} --frame_features=True \
                --feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 \
                --learning_rate_decay=0.8 --l2_penalty=1e-5 --max_step=700000 --num_gpu=2

python eval.py ${parameters} --batch_size=80 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5\
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/validate*.tfrecord --train_dir ${train_dir} \
               --run_once=True

mkdir -p $result_folder
python inference.py --output_model_tgz ${result_folder}/${train_dir}.tgz \
                    --output_file ${result_folder}/${train_dir}.csv \
                    --input_data_pattern=${test_path}/test*.tfrecord --train_dir ${train_dir} \
                    --batch_size=80 --num_readers=8

