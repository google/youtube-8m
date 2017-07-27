import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, learn, losses, metrics
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.training import HParams
# from tensorflow.python.training import basic_session_run_hooks as bhooks
import time
import itertools

import json

import eval_util
import export_model
import losses
import frame_level_models
import video_level_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
import utils

FLAGS = flags.FLAGS




tf.logging.set_verbosity(tf.logging.INFO)  # enables training error print out during training

if __name__ == '__main__':
    flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                        "The directory to save the model files in.")
    flags.DEFINE_string(
        "train_data_pattern", "",
        "File glob for the training dataset. If the files refer to Frame Level "
        "features (i.e. tensorflow.SequenceExample), then set --reader_type "
        "format. The (Sequence)Examples are expected to have 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                        "to use for training.")
    flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

    # Model flags.
    flags.DEFINE_bool(
        "frame_features", False,
        "If set, then --train_data_pattern must be frame-level features. "
        "Otherwise, --train_data_pattern must be aggregated video-level "
        "features. The model must also be set appropriately (i.e. to read 3D "
        "batches VS 4D batches.")
    flags.DEFINE_string(
        "model", "LogisticModel",
        "Which architecture to use for the model. Models are defined "
        "in models.py.")
    flags.DEFINE_bool(
        "start_new_model", False,
        "If set, this will not resume from a checkpoint and will instead create a"
        " new model instance.")

    # Training flags.
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                        "Which loss function to use for training the model.")
    flags.DEFINE_float(
        "regularization_penalty", 1.0,
        "How much weight to give to the regularization loss (the label loss has "
        "a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.01,
                       "Which learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.95,
                       "Learning rate decay factor to be applied every "
                       "learning_rate_decay_examples.")
    flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                       "Multiply current learning rate by learning_rate_decay "
                       "every learning_rate_decay_examples.")
    flags.DEFINE_integer("num_epochs", 5,
                         "How many passes to make over the dataset before "
                         "halting training.")
    flags.DEFINE_integer("max_steps", None,
                         "The maximum number of iterations of the training loop.")
    flags.DEFINE_integer("export_model_steps", 1000,
                         "The period, in number of steps, with which the model "
                         "is exported for batch prediction.")

    # Other flags.
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_string("optimizer", "AdamOptimizer",
                        "What optimizer class to use.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
    flags.DEFINE_bool(
        "log_device_placement", False,
        "Whether to write the device on which every op will run into the "
        "logs on startup.")
def model_fn(features,labels,mode,params):
    outputs = layers.fully_connected(
                    inputs = features,
                    num_outputs = 10,
                    activation_fn = None,
                    trainable = is_training,
                    scope = 'LayersDNN_Output')
    loss = None
    train_op = None
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(outputs, labels_norm)
        loss = tf.identity(loss,'MSE')

    train_op = tf.contrib.layers.optimize_loss(
      loss, None, optimizer='Adam',
                learning_rate = params.learning_rate)

    predictions = {"predictions":tf.identity(outputs,name = 'predictions')}
    return model_fn_lib.ModelFnOps( mode=mode, predictions=predictions,
                                    loss=loss, train_op=train_op)
def train_input_fn(params):
    xbatch = tf.random_normal([params.batch_size, 512], dtype=tf.float32)
    ybatch = tf.random_normal([params.batch_size, 256], dtype=tf.float32)
    return [xbatch,ybatch]
def _experiment_fn(run_config, hparams):
    # Create Estimator
     # seems to be the only way to stop CUDA_OUT_MEMORY_ERRORs
    estimator = learn.Estimator(model_fn=model_fn,
                                config=run_config, params=hparams)

    return tf.contrib.learn.Experiment(estimator=estimator,
                                       train_input_fn=lambda: train_input_fn(hparams),
                                       eval_input_fn=None,
                                       train_steps=5000,
                                       train_monitors = [stager_hook])
def main(argv=None):
    hparams = HParams(
                     batch_size = 128,
                     hidden_units=[256],
                     learning_rate = .001)

    output_dir= 'test'

    config = learn.RunConfig(save_checkpoints_secs = 600,
                             model_dir = output_dir,
                             gpu_memory_fraction=1)
    learn_runner.run(experiment_fn = _experiment_fn,
                      run_config = config,
                      hparams = hparams)
if __name__ == '__main__':

    tf.app.run()
