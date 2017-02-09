# YouTube-8M Tensorflow Starter Code

This repo contains starter code for training and evaluating machine learning
models over the [YouTube-8M](https://research.google.com/youtube8m/) dataset.
The code gives an end-to-end working example for reading the dataset, training a
TensorFlow model, and evaluating the performance of the model. Out of the box,
you can train a logistic classification model over either frame-level or
video-level features. The code can be extended to train more complex models.

It is possible to train and evaluate on YouTube-8M in two ways: on your own
machine, or on Google Cloud. This README provides instructions for both.


## Table of Contents
* [Running on Google's Cloud Machine Learning Platform](#running-on-googles-cloud-machine-learning-platform)
   * [Requirements](#requirements)
   * [Training on Video-Level Features](#training-on-video-level-features)
   * [Evaluation and Inference](#evaluation-and-inference)
   * [Using Frame-Level features](#using-frame-level-features)
   * [Using audio features](#using-audio-features)
   * [Testing Locally](#testing-locally)
* [Running on your Own Machine](#running-on-your-own-machine)
   * [Requirements](#requirements-1)
   * [Training on Video-Level features](#training-on-video-level-features-1)
   * [Evaluation and Inference](#evaluation-and-inference-1)
   * [Using Frame-Level Features](#using-frame-level-features-1)
   * [Using audio features](#using-audio-features-1)
* [Overview of Files](#overview-of-files)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Inference](#inference)
   * [Misc](#misc)
* [About this project](#about-this-project)

## Running on Google's Cloud Machine Learning Platform

### Requirements

This option only requires you to have an
[appropriately configured](https://cloud.google.com/ml/docs/how-tos/getting-set-up)
Google Cloud Platform account. Since you will be running code and accessing
data files in the cloud, you do not need to install any libraries or download
the training data. If you would like to be able to run Tensorboard or test
your code locally before deploying it to the cloud, see the
[Testing Locally](#testing-locally) section.

### Training on Video-Level Features

You can train over video-level features with a few commands. First, navigate to
the directory *immediately above* the source code. You should be able to see the
source code directory if you run 'ls'. Then run the following:

```sh
BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-central1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_`date +%s`; gcloud --verbosity=debug beta ml jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
-- --train_data_pattern='gs://youtube8m-ml/0/video_level/train/*.tfrecord' \
--train_dir=$BUCKET_NAME/$JOB_NAME
```

In the gsutil command above, the "package-path" flag refers to the directory
containing the "train.py" script and more generally the python package which
should be deployed to the cloud worker. The module-name refers to the specific
python script which should be executed (in this case the train module).

The training data files are hosted in the public "youtube8m-ml" storage bucket
in the "us-central1" region. Therefore, we've colocated our job in the same
region in order to have the fastest access to the data.

Generally, you should use a different $JOB_NAME for each of your individual
runs.

### Evaluation and Inference
Here's how to evaluate a model on the validation dataset:

```sh
JOB_TO_EVAL=yt8m_train
JOB_NAME=yt8m_eval_`date +%s`; gcloud --verbosity=debug beta ml jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
-- --eval_data_pattern='gs://youtube8m-ml/0/video_level/validate/*.tfrecord' \
--train_dir=$BUCKET_NAME/$JOB_TO_EVAL
```

And here's how to perform inference with a model:

```sh
JOB_TO_EVAL=yt8m_train
JOB_NAME=yt8m_inference_`date +%s`; gcloud --verbosity=debug beta ml jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.inference \
--staging-bucket=$BUCKET_NAME --region=us-central1 \
-- --input_data_pattern='gs://youtube8m-ml/0/video_level/validate/*.tfrecord' \
--train_dir=$BUCKET_NAME/$JOB_TO_EVAL \
--output_file=$BUCKET_NAME/$JOB_TO_EVAL/predictions.csv
```

Note the confusing use of "training" in the above gcloud commands. Despite the
name, the 'training' argument really just offers a cloud hosted
python/tensorflow service. From the point of view of the Cloud Platform, there
is no distinction between our training and inference jobs. The Cloud ML platform
also offers specialized functionality for prediction with
Tensorflow models, but discussing that is beyond the scope of this readme.

### Using Frame-Level features

Append
```sh
--frame_features=True --model=FrameLevelLogisticModel --feature_names=rgb \
--batch_size=256
```

to the 'gcloud' commands given above, and change 'video_level' in paths to
'frame_level'.

The 'FrameLevelLogisticModel' is designed to provide equivalent results to a
logistic model trained over the video-level features. Please look at the
'models.py' file to see how to implement your own models.


### Using audio features

The feature files (both Frame-Level and Video-Level) contain two sets of
features: 1) visual and 2) audio. The code defaults to using the visual
features only, but it is possible to use audio features instead of (or besides)
visual features. To specify the (combination of) features to use you must set
`--feature_names` and `feature_sizes` flags, which take a comma-separated list
of feature names and feature sizes, respectively. The visual and audio features
are named `rgb` and `audio` and have `1024` and `128` dimensions, respectively.
For example, to use audio-visual Video-Level features the following flags must
be set:
```
--feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128"
```

Similarly, to audio-visual Frame-Level features the following flags must be
set:
```
--feature_names="rgb, audio" --feature_sizes="1024, 128"
```

NOTE: make sure the set of features and the order in which the appear in the
lists provided to the two flags above match when running training, evaluation,
or inference.

### Testing Locally
As you are developing your own models, you might want to be able to test them
quickly without having to submit them to the cloud. You can use the
`gcloud beta ml local` set of commands for that. First, since you are running
locally you will need to install [Tensorflow](https://tensorflow.org).

Here is an example command line for video-level training:

```sh
gcloud --verbosity=debug beta ml local train \
--package-path=youtube-8m --module-name=youtube-8m.train -- \
--train_data_pattern='gs://youtube8m-ml/0/video_level/train/*.tfrecord' \
--train_dir=/tmp/yt8m_train --start_new_model
```

You can modify this template using the instructions above to train with
frame-level features or to do evaluation or inference. You might also want to
download some training shards locally, to speed things up and allow you to
work offline. The command below will copy 10 out of the 4096 training data files
to the current directory.

```sh
# Downloads 50MB of data.
gsutil cp gs://us.data.yt8m.org/0/video_level/train/traina[0-9].tfrecord .
```

Once you download the files, you can point the job to them using the
'train_data_pattern' argument.

By installing Tensorflow locally, you will also get access to the Tensorboard
tool, which allows you to view and compare metrics for your various models.
You can have Tensorboard read the data directly from your Cloud ML bucket

```sh
tensorboard --logdir=$BUCKET_NAME
```

## Running on your Own Machine

### Requirements

The starter code requires Tensorflow. If you haven't installed it yet, follow
the instructions on [tensorflow.org](https://tensorflow.org). This code has been
tested with Tensorflow version 0.12.0-rc1. Going forward, we will continue to
target the latest released version of Tensorflow.

You can download the YouTube-8M data files from
[here](https://research.google.com/youtube8m/download.html). We recommend
downloading the smaller video-level features dataset first when getting started.

### Training on Video-Level features

To start training a logistic model on the video-level features, run

```sh
MODEL_DIR=/tmp/yt8m
python train.py --train_data_pattern='/path/to/features/train*.tfrecord' --train_dir=$MODEL_DIR/logistic_model
```

Since the dataset is sharded into 4096 individual files, we use a wildcard (\*)
to represent all of those files.

By default, the training code will frequently write _checkpoint_ files (i.e.
values of all trainable parameters, at the current training iteration). These
will be written to the `--train_dir`. If you re-use a `--train_dir`, the trainer
will first restore the latest checkpoint written in that directory. This only
works if the architecture of the checkpoint matches the graph created by the
training code. If you are in active development/debugging phase, consider
adding `--start_new_model` flag to your run configuration.

### Evaluation and Inference

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

### Using Frame-Level Features

Follow the same instructions as above, appending
`--frame_features=True --model=FrameLevelLogisticModel --feature_names=rgb`
for the train.py, eval.py, and inference.py scripts.

The 'FrameLevelLogisticModel' is designed to provide equivalent results to a
logistic model trained over the video-level features. Please look at the
'models.py' file to see how to implement your own models.

### Using audio features

See [Using audio features](#using-audio-features) section above.

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
