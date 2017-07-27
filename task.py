import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, learn, losses, metrics
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
import hooks
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)  # enables training error print out during training

if __name__ == '__main__':
    flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                        "The directory to save the model files in.")
    flags.DEFINE_string(
        "eval_data_pattern", "",
        "File glob defining the evaluation dataset in tensorflow.SequenceExample "
        "format. The SequenceExamples are expected to have an 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
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

def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                             (category, flag_value,
                              expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def model_fn(features, labels, mode, params):


    optimizer_class = find_class_by_name(params.optimizer, [tf.train])
    label_loss_fn = find_class_by_name(params.label_loss, [losses])()
    model = find_class_by_name(params.model,
                               [frame_level_models, video_level_models])()

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
                    params.base_learning_rate,
                    global_step * params.batch_size * params.num_towers,
                    params.learning_rate_decay_examples,
                    params.learning_rate_decay,
                    staircase=True,
                    )

    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = optimizer_class(learning_rate)


    tf.summary.histogram("model/input_raw", features['model_input'])

    feature_dim = len(features['model_input'].get_shape()) - 1

    model_input = tf.nn.l2_normalize(features['model_input'], feature_dim)

    tower_inputs = tf.split(model_input, params.num_towers)
    tower_labels = tf.split(labels, params.num_towers)
    tower_num_frames = tf.split(features['num_frames'], params.num_towers)
    tower_gradients = []
    tower_predictions = []
    tower_label_losses = []
    tower_reg_losses = []




    for i in range(params.num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
        with tf.device(params.device_string % i):
            with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
                with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if params.num_gpus!=1 else "/gpu:0")):
                     result = model.create_model(
                       tower_inputs[i],
                       num_frames=tower_num_frames[i],
                       vocab_size=params.reader.num_classes,
                       labels=tower_labels[i])
                     for variable in slim.get_model_variables():
                       tf.summary.histogram(variable.op.name, variable)

                     predictions = result["predictions"]
                     tower_predictions.append(predictions)

                     if "loss" in result.keys():
                       label_loss = result["loss"]
                     else:
                       label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i])

                     if "regularization_loss" in result.keys():
                       reg_loss = result["regularization_loss"]
                     else:
                       reg_loss = tf.constant(0.0)

                     reg_losses = tf.losses.get_regularization_losses()
                     if reg_losses:
                       reg_loss += tf.add_n(reg_losses)

                     tower_reg_losses.append(reg_loss)



                     # Adds update_ops (e.g., moving average updates in batch normalization) as
                     # a dependency to the train_op.
                     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                     if "update_ops" in result.keys():
                       update_ops += result["update_ops"]
                     if update_ops:
                       with tf.control_dependencies(update_ops):
                         barrier = tf.no_op(name="gradient_barrier")
                         with tf.control_dependencies([barrier]):
                           label_loss = tf.identity(label_loss)

                     tower_label_losses.append(label_loss)

    label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
    predictions = tf.concat(tower_predictions, 0)
    tf.summary.scalar("label_loss", label_loss)
    if params.regularization_penalty != 0:
        reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
        tf.summary.scalar("reg_loss", reg_loss)

    if mode == learn.ModeKeys.TRAIN:
        # Incorporate the L2 weight penalties, etc.
        final_loss = params.regularization_penalty * reg_loss + label_loss
        gradients = optimizer.compute_gradients(
            final_loss, colocate_gradients_with_ops=False)
        tower_gradients.append(gradients)
        merged_gradients = utils.combine_gradients(tower_gradients)
        if params.clip_gradient_norm > 0:
            with tf.name_scope('clip_grads'):
                merged_gradients = utils.clip_gradient_norms(merged_gradients, params.clip_gradient_norm)
        train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)
    else:
        train_op = None

    eval_metric_ops = {}
    if mode == learn.ModeKeys.EVAL:

        eval_metric_ops['hit_at_one'] = metrics.streaming_mean(tf.py_func(lambda x,y: np.float32(eval_util.calculate_hit_at_one(x,y)),
                                                   [predictions, labels],
                                                   tf.float32,
                                                   stateful=False,
                                                   ))
        eval_metric_ops['perr'] = metrics.streaming_mean(tf.py_func(lambda x,y: np.float32(eval_util.calculate_precision_at_equal_recall_rate(x,y)),
                                             [predictions, labels],
                                             tf.float32,
                                             stateful=False,
                                             ))
        eval_metric_ops['gap'] = metrics.streaming_mean(tf.py_func(lambda x,y: np.float32(eval_util.calculate_gap(x,y)),
                                            [predictions, labels],
                                            tf.float32,
                                            stateful=False,
                                            ))

    #  tf.add_to_collection("global_step", global_step)
    #  tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
    #  tf.add_to_collection("input_batch_raw", model_input_raw)
    #  tf.add_to_collection("input_batch", model_input)
    #  tf.add_to_collection("num_frames", num_frames)
    tf.add_to_collection("labels", tf.cast(labels, tf.float32))
    #  tf.add_to_collection("train_op", train_op)
    tf.summary.scalar("loss", label_loss)
    return learn.ModelFnOps(mode=mode,
                            predictions=predictions,
                            loss=label_loss,
                            train_op=train_op,
                            eval_metric_ops=eval_metric_ops,
                            )

def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)

  return reader



def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' styxle path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.batch_join(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)

def train_input_fn(params):

    unused_video_id, model_input_raw, labels_batch, num_frames = (
        get_input_data_tensors(
            params.reader,
            params.train_data_pattern,
            batch_size=params.batch_size * params.num_towers,
            num_readers=params.num_readers,
            num_epochs=params.num_epochs))
    features = {}

    features['model_input'] = model_input_raw
    features['num_frames'] = num_frames
    return [features, labels_batch]

def eval_input_fn(params):
    video_id_batch, model_input_raw, labels_batch, num_frames = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
        params.reader,
        params.eval_data_pattern,
        batch_size=params.batch_size,
        num_readers=params.num_readers)
    features = {}

    features['model_input'] = model_input_raw
    features['num_frames'] = num_frames
    return [features, labels_batch]

def _experiment_fn(run_config, hparams):
    # Create Estimator
     # seems to be the only way to stop CUDA_OUT_MEMORY_ERRORs
    estimator = learn.Estimator(model_fn=model_fn,
                                config=run_config,
                                params=hparams,
                                )
    #eval_hook = hooks.EvalMetricsHook(FLAGS.train_dir)

    return tf.contrib.learn.Experiment(
            estimator=estimator,
            train_input_fn=lambda: train_input_fn(hparams),
            eval_input_fn=lambda: eval_input_fn(hparams),
            train_steps = 10000
            #train_monitors = [eval_hook]
            )


def main(argv=None):
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    num_gpus = len(gpus)

    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'

    hparams = HParams(
        batch_size=FLAGS.batch_size,
        learning_rate_decay=FLAGS.learning_rate_decay,
        learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
        optimizer=FLAGS.optimizer,
        label_loss=FLAGS.label_loss,
        model=FLAGS.model,
        base_learning_rate=FLAGS.base_learning_rate,
        reader=get_reader(),
        num_towers=num_towers,
        device_string=device_string,
        num_readers=FLAGS.num_readers,
        num_epochs=FLAGS.num_epochs,
        train_data_pattern=FLAGS.train_data_pattern,
        eval_data_pattern=FLAGS.eval_data_pattern,
        num_gpus = num_gpus,
        regularization_penalty=FLAGS.regularization_penalty,
        clip_gradient_norm = FLAGS.clip_gradient_norm)

    config = learn.RunConfig(save_checkpoints_secs=15 * 60,
                             save_summary_steps=1000,
                             model_dir=FLAGS.train_dir,
                             gpu_memory_fraction=1,
                             )
    learn_runner.run(experiment_fn = _experiment_fn,
                      run_config = config,
                      hparams = hparams,
                      schedule = 'train_and_evaluate')


def remove_training_directory(train_dir):
    """Removes the training directory."""
    try:
        logging.info("Removing existing train directory.")
        gfile.DeleteRecursively(train_dir)
    except:
        logging.error("Failed to delete directory " + train_dir
                      + " when starting a new model; please delete it "
                      + "manually and try again.")


def dist_setup():
    config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    task_env = config.get('task', {})
    task_type = None
    task_index = None
    if task_env:
        task_type = task_env.get('type').encode('ascii')
        task_index = task_env.get('index')
    return task_type, task_index


if __name__ == '__main__':
    task_type, _ = dist_setup()
    if task_type in [None, 'master'] and FLAGS.start_new_model:
        remove_training_directory(FLAGS.train_dir)
    tf.app.run()
