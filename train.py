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
  flags.DEFINE_integer(
      "batch_size", 1024,
      "How many examples to process per batch for training.")
  flags.DEFINE_string(
      "label_loss", "CrossEntropyLoss",
      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1,
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

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("master", "", "TensorFlow master to use.")
  flags.DEFINE_integer("task", 0, "Task id of the replica running the training."
                       " 0 implies chief Supervisor.""")
  flags.DEFINE_integer("ps_tasks", 0, """Number of tasks in the ps job.
                       If 0 no ps job is used.""")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")

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
                    data_pattern + "'")
    logging.info("number of training files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(files,
                                                    num_epochs=num_epochs)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=FLAGS.batch_size * 5,
        min_after_dequeue=FLAGS.batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


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
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=100):
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
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """
  with tf.device(tf.train.replica_device_setter(
      FLAGS.ps_tasks, merge_devices=True)):
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * batch_size,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = optimizer_class(learning_rate)
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
      result = model.create_model(model_input,
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
      reg_losses = tf.losses.get_regularization_losses()
      if reg_losses:
        reg_loss += tf.add_n(reg_losses)
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
    train_op = slim.learning.create_train_op(
        final_loss,
        optimizer,
        global_step=global_step,
        clip_gradient_norm=clip_gradient_norm)

    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", predictions)
    tf.add_to_collection("input_batch_raw", model_input_raw)
    tf.add_to_collection("input_batch", model_input)
    tf.add_to_collection("num_frames", num_frames)
    tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
    tf.add_to_collection("train_op", train_op)


def train_loop(train_dir=None,
               saver=None,
               is_chief=True,
               master="",
               start_supervisor_services=True):
  """Performs training on the currently defined tensorflow graph.

  Args:
    train_dir: Where to save the model checkpoints.
    saver: The class to use for serializing the graph variables.
    is_chief: Whether this worker is the primary worker (which is responsible
    for writing checkpoints and summaries), or an anonymous member of the flock.
    master: Which Tensorflow master to listen to.
    start_supervisor_services: Whether to start threads for writing summaries
      and checkpoints.

  Returns:
  A tuple of the training Hit@1 and the training PERR.
  """
  global_step = tf.get_collection("global_step")[0]
  loss = tf.get_collection("loss")[0]
  predictions = tf.get_collection("predictions")[0]
  labels = tf.get_collection("labels")[0]
  train_op = tf.get_collection("train_op")[0]

  sv = tf.train.Supervisor(logdir=train_dir,
                           is_chief=is_chief,
                           global_step=global_step,
                           save_model_secs=15 * 60,
                           save_summaries_secs=120,
                           saver=saver)
  sess = sv.prepare_or_wait_for_session(
      master,
      start_standard_services=start_supervisor_services,
      config=tf.ConfigProto(
          log_device_placement=False,
          allow_soft_placement=True))

  logging.info("prepared session")
  sv.start_queue_runners(sess)
  logging.info("started queue runners")

  try:
    logging.info("entering training loop")
    while not sv.should_stop():
      batch_start_time = time.time()
      _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
          [train_op, global_step, loss, predictions, labels])
      seconds_per_batch = time.time() - batch_start_time
      examples_per_second = labels_val.shape[0] / seconds_per_batch

      hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
      perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,
                                                                labels_val)
      gap = eval_util.calculate_gap(predictions_val, labels_val)

      logging.info("training step " + str(global_step_val) + "| Hit@1: " + (
          "%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) +
          " GAP: " + ("%.2f" % gap) + " Loss: " + str(loss_val))
      if is_chief and global_step_val % 10 == 0 and train_dir:
        sv.summary_writer.add_summary(
            utils.MakeSummary("model/Training_Hit@1",
                              hit_at_one), global_step_val)
        sv.summary_writer.add_summary(
            utils.MakeSummary("model/Training_Perr", perr),
            global_step_val)
        sv.summary_writer.add_summary(
            utils.MakeSummary("model/Training_GAP", gap),
            global_step_val)
        sv.summary_writer.add_summary(
            utils.MakeSummary("global_step/Examples/Second",
                              examples_per_second),
            global_step_val)
        sv.summary_writer.flush()
  except tf.errors.OutOfRangeError:
    logging.info("Done training -- epoch limit reached")
  logging.info("exited training loop")
  sv.Stop()
  return hit_at_one, perr


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  is_chief = (FLAGS.task == 0)

  # Recover session
  saver = None
  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
  if FLAGS.start_new_model:
    logging.info("'start_new_model' flag is set. Removing existing train dir.")
    try:
      gfile.DeleteRecursively(FLAGS.train_dir)
    except:
      logging.error(
          "Failed to delete directory " + FLAGS.train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.")
  elif not latest_checkpoint:
    logging.info("No checkpoint file found. Building a new model.")
  else:
    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("No meta graph file found. Building a new model.")
    else:
      logging.info("Restoring from meta graph file %s", meta_filename)
      saver = tf.train.import_meta_graph(meta_filename)

  if not saver:
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    if FLAGS.frame_features:
      reader = readers.YT8MFrameFeatureReader(
          feature_names=feature_names,
          feature_sizes=feature_sizes)
    else:
      reader = readers.YT8MAggregatedFeatureReader(
          feature_names=feature_names,
          feature_sizes=feature_sizes)

    model = find_class_by_name(FLAGS.model,
        [frame_level_models, video_level_models])()
    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])
    build_graph(reader=reader,
                model=model,
                optimizer_class=optimizer_class,
                clip_gradient_norm=FLAGS.clip_gradient_norm,
                train_data_pattern=FLAGS.train_data_pattern,
                label_loss_fn=label_loss_fn,
                base_learning_rate=FLAGS.base_learning_rate,
                learning_rate_decay=FLAGS.learning_rate_decay,
                learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                regularization_penalty=FLAGS.regularization_penalty,
                num_readers=FLAGS.num_readers,
                batch_size=FLAGS.batch_size,
                num_epochs=FLAGS.num_epochs)
    logging.info("built graph")
    saver = tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)

  train_loop(is_chief=is_chief,
             train_dir=FLAGS.train_dir,
             saver=saver,
             master=FLAGS.master)


if __name__ == "__main__":
  app.run()

