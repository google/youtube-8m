# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

import eval_util
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
import utils

_SAVE_INTERVAL_SECONDS = 60
_LOG_INTERVAL_SECONDS = 5

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
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
      "regularization_penalty", 1e-3,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
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
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." % (
          category, flag_value, expected_superclass.__name__))
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
        capacity=FLAGS.batch_size * 5,
        min_after_dequeue=FLAGS.batch_size,
        allow_smaller_final_batch=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                optimizer_class=tf.train.AdamOptimizer,
                regularization_penalty=1e-3,
                num_readers=1,
                num_epochs=None):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  optimizer = optimizer_class(base_learning_rate)
  unused_video_id, model_input_raw, labels_batch, num_frames = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size,
          num_readers=num_readers,
          num_epochs=num_epochs))
  tf.summary.histogram("model/input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  with tf.name_scope("model"):
    result = model.create_model(
        model_input,
        num_frames=num_frames,
        vocab_size=reader.num_classes,
        labels=labels_batch)

    for variable in slim.get_model_variables():
      tf.summary.histogram(variable.op.name, variable)

    predictions = result["predictions"]
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)
    tf.summary.scalar("label_loss", label_loss)

    if "regularization_loss" in result.keys():
      reg_loss = result["regularization_loss"]
    else:
      reg_loss = tf.constant(0.0)
    if regularization_penalty != 0:
      tf.summary.scalar("reg_loss", reg_loss)

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

    # Incorporate the L2 weight penalties etc.
    final_loss = regularization_penalty * reg_loss + label_loss
    train_op = optimizer.minimize(final_loss, global_step=global_step)

    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", predictions)
    tf.add_to_collection("input_batch_raw", model_input_raw)
    tf.add_to_collection("input_batch", model_input)
    tf.add_to_collection("num_frames", num_frames)
    tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
    tf.add_to_collection("train_op", train_op)


class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.log_device_placement = FLAGS.log_device_placement
    self.train_dir = FLAGS.train_dir
    self.config = tf.ConfigProto(log_device_placement=self.log_device_placement)

    if self.is_master and self.task.index > 0:
      raise StandardError("%s: Only one replica of master expected",
                          task_as_string(self.task))

  def run(self):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """

    self.remove_existing_training_directory()

    target, device_fn = self.start_server_if_distributed()

    with tf.Graph().as_default() as graph:
      with tf.device(device_fn):

        saver = self.recover_or_build_model()
        global_step = tf.get_collection("global_step")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        train_op = tf.get_collection("train_op")[0]
        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=60,
        save_summaries_secs=60,
        saver=saver)

    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with sv.managed_session(target, config=self.config) as sess:

      self.last_save = 0
      self.last_log = 0

      try:
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        while not sv.should_stop():

          batch_start_time = time.time()
          _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
              [train_op, global_step, loss, predictions, labels])
          seconds_per_batch = time.time() - batch_start_time

          self.now = time.time()
          is_time_to_save = (self.now - self.last_save) > _SAVE_INTERVAL_SECONDS
          is_time_to_log = (self.now - self.last_log) > _LOG_INTERVAL_SECONDS
          should_save = self.is_master and is_time_to_save and self.train_dir
          should_log = is_time_to_log or should_save

          if should_log or should_save:
            examples_per_second = labels_val.shape[0] / seconds_per_batch
            hit_at_one = eval_util.calculate_hit_at_one(predictions_val,
                                                        labels_val)
            perr = eval_util.calculate_precision_at_equal_recall_rate(
                predictions_val, labels_val)
            gap = eval_util.calculate_gap(predictions_val, labels_val)

          if should_log:
            logging.info(
                "%s: training step " + str(global_step_val) + "| Hit@1: " +
                ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) + " GAP: " +
                ("%.2f" % gap) + " Loss: " + str(loss_val),
                task_as_string(self.task))
            self.last_log = self.now

          if should_save:
            logging.info("%s: Writing summary.", task_as_string(self.task))
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Perr", perr), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_GAP", gap), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("global_step/Examples/Second",
                                  examples_per_second), global_step_val)
            sv.summary_writer.flush()
            self.last_save = self.now

      except tf.errors.OutOfRangeError:
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()
    return hit_at_one, perr

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_existing_training_directory(self):
    """Removes the training directory if requested."""

    if self.is_master and FLAGS.start_new_model:
      try:
        logging.info(
            "%s: Flag 'start_new_model' is set. Removing existing train directory.",
            task_as_string(self.task))
        gfile.DeleteRecursively(FLAGS.train_dir)
      except:
        logging.error(
            "%s: Failed to delete directory " + FLAGS.train_dir +
            " when starting a new model. Please delete it manually and" +
            " try again.", task_as_string(self.task))

  def recover_or_build_model(self):
    """Recovers the model from a checkpoint or build it."""

    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)

    if FLAGS.start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return self.find_and_build_model()
    elif not latest_checkpoint:
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return self.find_and_build_model()
    else:
      meta_filename = latest_checkpoint + ".meta"
      if not gfile.Exists(meta_filename):
        logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
        return self.find_and_build_model()
      else:
        logging.info("%s: Restoring from meta graph file %s",
                     task_as_string(self.task), meta_filename)
        return tf.train.import_meta_graph(meta_filename)

  def find_and_build_model(self):
    """Find the model and build the graph."""

    # Convert feature_names and feature_sizes to lists of values.
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    if FLAGS.frame_features:
      reader = readers.YT8MFrameFeatureReader(
          feature_names=feature_names, feature_sizes=feature_sizes)
    else:
      reader = readers.YT8MAggregatedFeatureReader(
          feature_names=feature_names, feature_sizes=feature_sizes)

    # Find the model.
    model = find_class_by_name(FLAGS.model,
                               [frame_level_models, video_level_models])()
    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    # Build the graph.
    build_graph(
        reader=reader,
        model=model,
        optimizer_class=optimizer_class,
        train_data_pattern=FLAGS.train_data_pattern,
        label_loss_fn=label_loss_fn,
        base_learning_rate=FLAGS.base_learning_rate,
        regularization_penalty=FLAGS.regularization_penalty,
        num_readers=FLAGS.num_readers,
        batch_size=FLAGS.batch_size)
    logging.info("%s: Built graph.", task_as_string(self.task))

    return tf.train.Saver()


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)


def dispatch(cluster, task):
  """Starts a Trainer or a ParameterServer."""

  if not cluster or task.type == "master" or task.type == "worker":
    Trainer(cluster, task).run()
  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))


def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)


def main(unused_argv):

  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  dispatch(cluster, task)


if __name__ == "__main__":
  app.run()
