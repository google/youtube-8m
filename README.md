# YouTube-8M Tensorflow Starter Code

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

## Using Frame Level Features

Follow the same instructions as above, appending
`--frame_features=True --model=FrameLevelLogisticModel --feature_names=inc3`
for the train.py and eval.py scripts.

The 'FrameLevelLogisticModel' is designed to provide equivalent results to a
logistic model trained over the video-level features. Please look at the
'models.py' file to see how to implement your own models.

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
