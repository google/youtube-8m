# YouTube-8M Tensorflow Starter Code

This repo contains starter code for training and evaluating machine learning
models over the [YouTube-8M](https://research.google.com/youtube8m/) dataset.
This is the starter code for our
[3rd Youtube8M Video Understanding Challenge on Kaggle](https://www.kaggle.com/c/youtube8m-2019)
and part of the International Conference on Computer Vision (ICCV) 2019 selected
workshop session. The code gives an end-to-end working example for reading the
dataset, training a TensorFlow model, and evaluating the performance of the
model.

## Table of Contents

*   [Running on Your Own Machine](#running-on-your-own-machine)
    *   [Requirements](#requirements)
    *   [Download Dataset Locally](#download-dataset-locally)
    *   [Try the starter code](#try-the-starter-code)
        *   [Train video-level model on frame-level features and inference at
            segment-level.](#train-video-level-model-on-frame-level-features-and-inference-at-segment-level)
        *   [Tensorboard](#tensorboard)
        *   [Using GPUs](#using-gpus)
*   [Running on Google's Cloud Machine Learning Platform](#running-on-googles-cloud-machine-learning-platform)
    *   [Requirements](#requirements-1)
    *   [Accessing Files on Google Cloud](#accessing-files-on-google-cloud)
    *   [Testing Locally](#testing-locally)
    *   [Training on the Cloud over Frame-Level Features](#training-on-the-cloud-over-frame-level-features)
    *   [Evaluation and Inference](#evaluation-and-inference)
*   [Create Your Own Dataset Files](#create-your-own-dataset-files)
*   [Training without this Starter Code](#training-without-this-starter-code)
*   [Export Your Model for MediaPipe Inference](#export-your-model-for-mediapipe-inference)
*   [More Documents](#more-documents)
*   [About This Project](#about-this-project)

## Running on Your Own Machine

### Requirements

The starter code requires Tensorflow. If you haven't installed it yet, follow
the instructions on [tensorflow.org](https://www.tensorflow.org/install/). This
code has been tested with Tensorflow 1.14. Going forward, we will continue to
target the latest released version of Tensorflow.

Please verify that you have Python 3.6+ and Tensorflow 1.14 or higher installed
by running the following commands:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

### Download Dataset Locally

Please see our
[dataset website](https://research.google.com/youtube8m/download.html) for
up-to-date download instructions.

In this document, we assume you download all the frame-level feature dataset to
`~/yt8m/2/frame` and segment-level validation/test dataset to `~/yt8m/3/frame`.
So the structure should look like

```
~/yt8m/
 - ~/yt8m/2/frame/
   - ~/yt8m/2/frame/train
 - ~/yt8m/3/frame/
   - ~/yt8m/3/frame/test
   - ~/yt8m/3/frame/validate
```

### Try the starter code

Clone this git repo: `mkdir -p ~/yt8m/code cd ~/yt8m/code git clone
https://github.com/google/youtube-8m.git`

#### Train video-level model on frame-level features and inference at segment-level.

Train using `train.py`, selecting a frame-level model (e.g.
`FrameLevelLogisticModel`), and instructing the trainer to use
`--frame_features`. TLDR - frame-level features are compressed, and this flag
uncompresses them.

```bash
python train.py --frame_features --model=FrameLevelLogisticModel \
--feature_names='rgb,audio' --feature_sizes='1024,128' \
--train_data_pattern=${HOME}/yt8m/2/frame/train/train*.tfrecord
--train_dir ~/yt8m/models/frame/sample_model --start_new_model
```

Evaluate the model by

```bash
python eval.py \
--eval_data_pattern=${HOME}/yt8m/3/frame/validate/validate*.tfrecord \
--train_dir ~/yt8m/models/frame/sample_model --segment_labels
```

This will provide some comprehensive metrics, e.g., gAP, mAP, etc., for your
models.

Produce CSV (`kaggle_solution.csv`) by doing inference:

```bash
python \
inference.py --train_dir ~/yt8m/models/frame/sample_model \
--output_file=$HOME/tmp/kaggle_solution.csv \
--input_data_pattern=${HOME}/yt8m/3/frame/test/test*.tfrecord --segment_labels
```

(Optional) If you wish to see how the models are evaluated in Kaggle system, you
can do so by

```bash
python inference.py --train_dir ~/yt8m/models/frame/sample_model \
--output_file=$HOME/tmp/kaggle_solution_validation.csv \
--input_data_pattern=${HOME}/yt8m/3/frame/validate/validate*.tfrecord \
--segment_labels
```

```bash
python segment_eval_inference.py \
--eval_data_pattern=${HOME}/yt8m/3/frame/validate/validate*.tfrecord \
--label_cache=$HOME/tmp/validate.label_cache \
--submission_file=$HOME/tmp/kaggle_solution_validation.csv --top_n=100000
```

**NOTE**: This script can be slow for the first time running. It will read
TFRecord data and build label cache. Once label cache is built, the evaluation
will be much faster later on.

#### Tensorboard

You can use Tensorboard to compare your frame-level or video-level models, like:

```sh
MODELS_DIR=~/yt8m/models
tensorboard --logdir frame:${MODELS_DIR}/frame
```

We find it useful to keep the tensorboard instance always running, as we train
and evaluate different models.

#### Using GPUs

If your Tensorflow installation has GPU support, e.g., installed with `pip
install tensorflow-gpu`, this code will make use of all of your compatible GPUs.
You can verify your installation by running

```
python -c 'import tensorflow as tf; tf.Session()'
```

This will print out something like the following for each of your compatible
GPUs.

```
I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties:
name: Tesla M40
major: 5 minor: 2 memoryClockRate (GHz) 1.112
pciBusID 0000:04:00.0
Total memory: 11.25GiB
Free memory: 11.09GiB
...
```

If at least one GPU was found, the forward and backward passes will be computed
with the GPUs, whereas the CPU will be used primarily for the input and output
pipelines. If you have multiple GPUs, the current default behavior is to use
only one of them.


## Running on Google's Cloud Machine Learning Platform

### Requirements

This option requires you to have an appropriately configured Google Cloud
Platform account. To create and configure your account, please make sure you
follow the instructions
[here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

Please also verify that you have Python 3.6+ and Tensorflow 1.14 or higher
installed by running the following commands:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

### Accessing Files on Google Cloud

You can browse the storage buckets you created on Google Cloud, for example, to
access the trained models, prediction CSV files, etc. by visiting the
[Google Cloud storage browser](https://console.cloud.google.com/storage/browser).

Alternatively, you can use the 'gsutil' command to download the files directly.
For example, to download the output of the inference code from the previous
section to your local machine, run:

```
gsutil cp $BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv .
```

### Testing Locally

All gcloud commands should be done from the directory *immediately above* the
source code. You should be able to see the source code directory if you run
'ls'.

As you are developing your own models, you will want to test them quickly to
flush out simple problems without having to submit them to the cloud.

Here is an example command line for frame-level training:

```sh
gcloud ai-platform local train \
--package-path=youtube-8m --module-name=youtube-8m.train -- \
--train_data_pattern='gs://youtube8m-ml/2/frame/train/train*.tfrecord' \
--train_dir=/tmp/yt8m_train --frame_features --model=FrameLevelLogisticModel \
--feature_names='rgb,audio' --feature_sizes='1024,128' --start_new_model
```

### Training on the Cloud over Frame-Level Features

The following commands will train a model on Google Cloud over frame-level
features.

```bash
BUCKET_NAME=gs://${USER}_yt8m_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml/2/frame/train/train*.tfrecord' \
--frame_features --model=FrameLevelLogisticModel \
--feature_names='rgb,audio' --feature_sizes='1024,128' \
--train_dir=$BUCKET_NAME/yt8m_train_frame_level_logistic_model --start_new_model
```

In the 'gsutil' command above, the 'package-path' flag refers to the directory
containing the 'train.py' script and more generally the python package which
should be deployed to the cloud worker. The module-name refers to the specific
python script which should be executed (in this case the train module).

It may take several minutes before the job starts running on Google Cloud. When
it starts you will see outputs like the following:

```
training step 270| Hit@1: 0.68 PERR: 0.52 Loss: 638.453
training step 271| Hit@1: 0.66 PERR: 0.49 Loss: 635.537
training step 272| Hit@1: 0.70 PERR: 0.52 Loss: 637.564
```

At this point you can disconnect your console by pressing "ctrl-c". The model
will continue to train indefinitely in the Cloud. Later, you can check on its
progress or halt the job by visiting the
[Google Cloud ML Jobs console](https://console.cloud.google.com/ml/jobs).

You can train many jobs at once and use tensorboard to compare their performance
visually.

```sh
tensorboard --logdir=$BUCKET_NAME --port=8080
```

Once tensorboard is running, you can access it at the following url:
[http://localhost:8080](http://localhost:8080). If you are using Google Cloud
Shell, you can instead click the Web Preview button on the upper left corner of
the Cloud Shell window and select "Preview on port 8080". This will bring up a
new browser tab with the Tensorboard view.

### Evaluation and Inference

Here's how to evaluate a model on the validation dataset:

```sh
JOB_TO_EVAL=yt8m_train_frame_level_logistic_model
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml/3/frame/validate/validate*.tfrecord' \
--frame_features --model=FrameLevelLogisticModel --feature_names='rgb,audio' \
--feature_sizes='1024,128' --train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--segment_labels --run_once=True
```

And here's how to perform inference with a model on the test set:

```sh
JOB_TO_EVAL=yt8m_train_frame_level_logistic_model
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ai-platform jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --input_data_pattern='gs://youtube8m-ml/3/frame/test/test*.tfrecord' \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --segment_labels \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv
```

Note the confusing use of 'training' in the above gcloud commands. Despite the
name, the 'training' argument really just offers a cloud hosted
python/tensorflow service. From the point of view of the Cloud Platform, there
is no distinction between our training and inference jobs. The Cloud ML platform
also offers specialized functionality for prediction with Tensorflow models, but
discussing that is beyond the scope of this readme.

Once these job starts executing you will see outputs similar to the following
for the evaluation code:

```
examples_processed: 1024 | global_step 447044 | Batch Hit@1: 0.782 | Batch PERR: 0.637 | Batch Loss: 7.821 | Examples_per_sec: 834.658
```

and the following for the inference code:

```
num examples processed: 8192 elapsed seconds: 14.85
```

## Export Your Model for MediaPipe Inference
To run inference with your model in [MediaPipe inference
demo](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/youtube8m#steps-to-run-the-youtube-8m-inference-graph-with-the-yt8m-dataset), you need to export your checkpoint to a SavedModel.

Example command:
```sh
python export_model_mediapipe.py --checkpoint_file  ~/yt8m/models/frame/sample_model/inference_model/segment_inference_model --output_dir /tmp/mediapipe/saved_model/
```


## Create Your Own Dataset Files

You can create your dataset files from your own videos. Our
[feature extractor](./feature_extractor) code creates `tfrecord` files,
identical to our dataset files. You can use our starter code to train on the
`tfrecord` files output by the feature extractor. In addition, you can fine-tune
your YouTube-8M models on your new dataset.

## Training without this Starter Code

You are welcome to use our dataset without using our starter code. However, if
you'd like to compete on Kaggle, then you must make sure that you are able to
produce a prediction CSV file produced by our `inference.py`. In particular, the
[predictions CSV file](https://www.kaggle.com/c/youtube8m-2018#evaluation) must
have two fields: `Class Id,Segment Ids` where `Class Id` must be class ids
listed in `segment_label_ids.csv` and `Segment Ids` is a space-delimited list of
`<video ID>:<segment start time>` **sorted in a descending order of confidence
score**.

Examples:

```
3,6l0e:100 6l0e:115 5x0Q:120 6l0e:160 Au0e:185 ...
...
1831,mm0Z:15 pT0e:190 sO0k:145 WQ01:195 Qd0K:175 ...
```

## More Documents

More documents can be found in [docs](./docs) folder.

## About This Project

This project is meant help people quickly get started working with the
[YouTube-8M](https://research.google.com/youtube8m/) dataset. This is not an
official Google product.
