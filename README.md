# YouTube-8M Tensorflow Starter Code

This repo contains starter code for training and evaluating machine learning
models over the [YouTube-8M](https://research.google.com/youtube8m/) dataset.
Out of the box, you can train a logistic classification model over either
frame-level or video-level features. The code can be easily extended to train
more sophisticated models.

## Requirements

This starter code requires Tensorflow. If you haven't installed it yet, follow
the instructions on [tensorflow.org](https://tensorflow.org). This code has been
tested with Tensorflow version 0.12.0-rc1. Going forward, we will continue to
target the latest released version of Tensorflow.

You can download the YouTube-8M data files from
[here](https://research.google.com/youtube8m/download.html). We recommend
downloading the smaller video-level features dataset first when getting started.

## Quick Start on Video-Level Features

To start training a logistic model on the video-level features, run

```sh
MODEL_DIR=/tmp/yt8m
python train.py --train_data_pattern='/path/to/features/train*.tfrecord' --train_dir=$MODEL_DIR/logistic_model
```

Since the dataset is sharded into 4096 individual files, we use a wildcard (\*)
to represent all of those files.

To evaluate the model, run

```sh
python eval.py --eval_data_pattern='/path/to/features/validate*.tfrecord' --train_dir=$MODEL_DIR/logistic_model
```

As the model is training or evaluating, you can view the results on tensorboard
by running

```sh
tensorboard --logdir=$MODEL_DIR
```

and navigating to http://localhost:6006 in your web browser.

When you are happy with your model, you can generate a csv file of predictions
from it by running

```sh
python inference.py --output_file=predictions.csv --input_data_pattern='/path/to/features/validate*.tfrecord' --train_dir=$MODEL_DIR/logistic_model
```

This will output the top 20 predicted labels from the model for every example to
'predictions.csv'.

## Using Frame Level Features

Follow the same instructions as above, appending
`--frame_features=True --model=FrameLevelLogisticModel --feature_names=inc3`
for the train.py, eval.py, and inference.py scripts.

The 'FrameLevelLogisticModel' is designed to provide equivalent results to a
logistic model trained over the video-level features. Please look at the
'models.py' file to see how to implement your own models.

## Training on Google Cloud Machine Learning Platform

This starter code is compatible with Google's Cloud Machine Learning Platform,
which is currently in beta..
We recommend initially testing your code at a small scale locally, and then
running your full scale training jobs in the cloud. This way, you avoid having
to download and store the full dataset, and you get access to high spec machines
for training.

After you've configured the Cloud SDK, you can train over frame-level features
with the following commands:
```sh
JOB_NAME=yt8m_train
BUCKET_NAME=gs://yt8m_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-central1 $BUCKET_NAME
# Submit the training job.
gcloud --verbosity=debug beta ml jobs submit training $JOB_NAME --package-path=youtube-8m --module-name=youtube-8m.train --staging-bucket=$BUCKET_NAME --region=us-central1 -- --train_data_pattern='gs://youtube8m-ml/0/train/*.tfrecord' --train_dir=$BUCKET_NAME/$JOB_NAME  --frame_features=True --model=FrameLevelLogisticModel --feature_names=inc3 --batch_size=256
```
In the command above, the "package-path" flag refers to the directory
containing the "train.py" script and more generally the python package which
should be deployed to the cloud worker. The module-name refers to the specific
python script which should be executed (in this case the train module).

The training data files are hosted in the public "youtube8m-ml" storage bucket
in the 'us-central1' region. Therefore, we've colocated our job in the same
region in order to have the fastest access to the data.

## Notes
One important thing to note is that by default, the train job will try to resume
from an existing model checkpoint if there is one in the training directory.
This may not be the behavior you want, especially during development.
To start training a fresh model, add the `--start_new_model` flag to your
run configuration.

## Overview of Files

### Training
*   `train.py`: The primary script for training models.
*   `losses.py`: Contains definitions for loss functions.
*   `models.py`: Contains definitions for models.
*   `readers.py`: Contains definitions for the Video dataset and Frame
                  dataset readers.

### Evaluation
*   `eval.py`: The primary script for evaluating models.
*   `eval_util.py`: Provides a class that calculates all evaluation metrics.
*   `average_precision_calculator.py`: Functions for calculating
                                       average precision.
*   `mean_average_precision_calculator.py`: Functions for calculating mean
                                            average precision.

### Inference
*   `inference.py`: Generates an output file containing predictions of
                    the model over a set of videos.

### Misc
*   `README.md`: This documentation.
*   `utils.py`: Common functions.

## About this project
This project is meant help people quickly get started working with the
[YouTube-8M](https://research.google.com/youtube8m/) dataset.
This is not an official Google product.
