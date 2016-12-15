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

import os
import time

import numpy
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils
import models

FLAGS = flags.FLAGS

if __name__ == '__main__':
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string("eval_dir", "/tmp/yt8m_eval/",
                      "The directory to save the evaluation results.")
  flags.DEFINE_string(
      "eval_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'INC6' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_name", "mean_inc3", "Name of the feature column "
                      "to use for training");
  flags.DEFINE_integer("feature_size", 1024, "length of the feature vectors");

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model. Options include 'Logistic', "
      "'SingleMixtureMoe', and 'TwoLayerSigmoid'. See aggregated_models.py and "
      "frame_level_models.py for the model definitions.")

  flags.DEFINE_integer(
      "batch_size", 1024,
      "How many examples to process per batch.")
  flags.DEFINE_integer("num_eval_examples", 1698333,
                       "Number of examples in the validation set.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Loss computed on validation data")
  flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")

  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_epochs=None,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

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
  tf.set_random_seed(0)  # must fix the seed for evaluation tensors.
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(files,
                                                    num_epochs=num_epochs,
                                                    shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in xrange(num_readers)]

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    video_id_batch, video_batch, labels_batch, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            capacity=capacity))
    tf.histogram_summary("video_batch", video_batch)
    return video_id_batch, video_batch, labels_batch, num_frames_batch


def build_graph(reader,
                model,
                eval_data_pattern,
                label_loss,
                batch_size=1024,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  video_id_batch, model_input_raw, labels_batch, num_frames = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
      reader,
      eval_data_pattern,
      batch_size=batch_size,
      num_readers=num_readers,
      num_epochs=None)

  feature_dim = len(model_input_raw.get_shape()) - 1
  feature_size = model_input_raw.get_shape()[feature_dim]

  # Normalize input features.
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  with tf.name_scope("model"):
    result = model.create_model(model_input,
                                num_frames=num_frames,
                                vocab_size=reader.num_classes,
                                labels=labels_batch,
                                is_training=False)
    predictions = result["predictions"]
    tf.histogram_summary("model_activations", predictions)
    if "loss" in result.keys():
      label_loss_val = result["loss"]
    else:
      label_loss_val = label_loss.calculate_loss(predictions, labels_batch)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss_val)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("summary_op", tf.merge_all_summaries())


def evaluation_loop(video_id_batch, prediction_batch, label_batch, loss,
                    summary_op, saver, summary_writer, evl_metrics,
                    last_global_step_val):
  """Run the evaluation loop once.

  Args:
    video_id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    if latest_checkpoint:
      logging.info("Loading checkpoint for eval: " + latest_checkpoint)
      # Restores from checkpoint
      saver.restore(sess, latest_checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = latest_checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info("skip this checkpoint global_step_val=%s "
                   "(same as the previous one).", global_step_val)
      return global_step_val

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch, label_batch, loss, summary_op]
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      max_eval_step = int(
          numpy.ceil(FLAGS.num_eval_examples / float(FLAGS.batch_size)))
      current_step = 0
      logging.info("enter eval_once loop global_step_val = %s. "
                   "Max step to run = %d", global_step_val, max_eval_step)

      evl_metrics.clear()  # clear the metrics

      while current_step < max_eval_step and not coord.should_stop():
        batch_start_time = time.time()
        _, predictions_val, labels_val, loss_val, summary_val = sess.run(
            fetches)
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = labels_val.shape[0] / seconds_per_batch

        iteration_info_dict = evl_metrics.accumulate(
            predictions_val, labels_val, loss_val,
            accumulate_average_precision=True)
        iteration_info_dict["examples_per_second"] = example_per_second

        if not FLAGS.run_once:
          iterinfo = utils.AddGlobalStepSummary(summary_writer,
                                                global_step_val,
                                                iteration_info_dict,
                                                summary_scope="Eval")
          logging.info("eval_step: %d | %s", current_step, iterinfo)
        current_step += 1

      # calculating the metrics for the entire epoch
      if not FLAGS.sample:
        epoch_info_dict = evl_metrics.get(FLAGS.num_eval_examples)
        epoch_info_dict["epoch_id"] = global_step_val

        summary_writer.add_summary(summary_val, global_step_val)
        epochinfo = utils.AddEpochSummary(summary_writer,
                                          global_step_val,
                                          epoch_info_dict,
                                          summary_scope="Eval")
        logging.info("--- " + epochinfo)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def evaluate():
  """Evaluate for a number of steps."""
  with tf.Graph().as_default():
    if FLAGS.frame_features:
      reader = readers.YT8MFrameFeatureReader(feature_name=FLAGS.feature_name,
                                              feature_size=FLAGS.feature_size)
    else:
      reader = readers.YT8MAggregatedFeatureReader(
          feature_name=FLAGS.feature_name, feature_size=FLAGS.feature_size)

    model = find_class_by_name(FLAGS.model, [models])()
    label_loss = find_class_by_name(FLAGS.label_loss, [losses])()

    build_graph(reader=reader,
                model=model,
                eval_data_pattern=FLAGS.eval_data_pattern,
                label_loss=label_loss,
                num_readers=FLAGS.num_readers,
                batch_size=FLAGS.batch_size)
    logging.info("built evaluation graph")
    video_id_batch = tf.get_collection("video_id_batch")[0]
    prediction_batch = tf.get_collection("predictions")[0]
    label_batch = tf.get_collection("labels")[0]
    loss = tf.get_collection("loss")[0]
    summary_op = tf.get_collection("summary_op")[0]

    saver = tf.train.Saver(tf.all_variables())
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph=tf.get_default_graph())

    evl_metrics = (eval_util.EvaluationMetrics(
        reader.num_classes, [10000 for _ in range(reader.num_classes)]))

    last_global_step_val = -1
    while True:
      last_global_step_val = evaluation_loop(video_id_batch, prediction_batch,
                                             label_batch, loss, summary_op,
                                             saver, summary_writer, evl_metrics,
                                             last_global_step_val)
      if FLAGS.run_once:
        break


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  if not gfile.Exists(FLAGS.eval_dir):
    gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == "__main__":
  app.run()
