# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Binary for generating predictions over a set of videos."""

from __future__ import print_function

import glob
import heapq
import json
import os
import tarfile
import tempfile
import time
import numpy as np

import readers
from six.moves import urllib
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.lib.io import file_io
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Input
  flags.DEFINE_string(
      "train_dir", "", "The directory to load the model files from. We assume "
      "that you have already run eval.py onto this, such that "
      "inference_model.* files already exist.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string(
      "input_model_tgz", "",
      "If given, must be path to a .tgz file that was written "
      "by this binary using flag --output_model_tgz. In this "
      "case, the .tgz file will be untarred to "
      "--untar_model_dir and the model will be used for "
      "inference.")
  flags.DEFINE_string(
      "untar_model_dir", "/tmp/yt8m-model",
      "If --input_model_tgz is given, then this directory will "
      "be created and the contents of the .tgz file will be "
      "untarred here.")
  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --input_data_pattern must be frame-level features (but with"
      " segment_labels). Otherwise, --input_data_pattern must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")
  flags.DEFINE_integer("segment_max_pred", 100000,
                       "Limit total number of segment outputs per entity.")
  flags.DEFINE_string(
      "segment_label_ids_file",
      "https://raw.githubusercontent.com/google/youtube-8m/master/segment_label_ids.csv",
      "The file that contains the segment label ids.")

  # Output
  flags.DEFINE_string("output_file", "", "The file to save the predictions to.")
  flags.DEFINE_string(
      "output_model_tgz", "",
      "If given, should be a filename with a .tgz extension, "
      "the model graph and checkpoint will be bundled in this "
      "gzip tar. This file can be uploaded to Kaggle for the "
      "top 10 participants.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

  # Other flags.
  flags.DEFINE_integer("batch_size", 512,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")


def format_lines(video_ids, predictions, top_k, whitelisted_cls_mask=None):
  """Create an information line the submission file."""
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    video_prediction = predictions[video_index]
    if whitelisted_cls_mask is not None:
      # Whitelist classes.
      video_prediction *= whitelisted_cls_mask
    top_indices = np.argpartition(video_prediction, -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield (video_ids[video_index] + "," +
           " ".join("%i %g" % (label, score) for (label, score) in line) +
           "\n").encode("utf8")


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(files,
                                                    num_epochs=1,
                                                    shuffle=False)
    examples_and_labels = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    input_data_dict = (tf.train.batch_join(examples_and_labels,
                                           batch_size=batch_size,
                                           allow_smaller_final_batch=True,
                                           enqueue_many=True))
    video_id_batch = input_data_dict["video_ids"]
    video_batch = input_data_dict["video_matrix"]
    num_frames_batch = input_data_dict["num_frames"]
    return video_id_batch, video_batch, num_frames_batch


def get_segments(batch_video_mtx, batch_num_frames, segment_size):
  """Get segment-level inputs from frame-level features."""
  video_batch_size = batch_video_mtx.shape[0]
  max_frame = batch_video_mtx.shape[1]
  feature_dim = batch_video_mtx.shape[-1]
  padded_segment_sizes = (batch_num_frames + segment_size - 1) // segment_size
  padded_segment_sizes *= segment_size
  segment_mask = (
      0 < (padded_segment_sizes[:, np.newaxis] - np.arange(0, max_frame)))

  # Segment bags.
  frame_bags = batch_video_mtx.reshape((-1, feature_dim))
  segment_frames = frame_bags[segment_mask.reshape(-1)].reshape(
      (-1, segment_size, feature_dim))

  # Segment num frames.
  segment_start_times = np.arange(0, max_frame, segment_size)
  num_segments = batch_num_frames[:, np.newaxis] - segment_start_times
  num_segment_bags = num_segments.reshape((-1))
  valid_segment_mask = num_segment_bags > 0
  segment_num_frames = num_segment_bags[valid_segment_mask]
  segment_num_frames[segment_num_frames > segment_size] = segment_size

  max_segment_num = (max_frame + segment_size - 1) // segment_size
  video_idxs = np.tile(
      np.arange(0, video_batch_size)[:, np.newaxis], [1, max_segment_num])
  segment_idxs = np.tile(segment_start_times, [video_batch_size, 1])
  idx_bags = np.stack([video_idxs, segment_idxs], axis=-1).reshape((-1, 2))
  video_segment_ids = idx_bags[valid_segment_mask]

  return {
      "video_batch": segment_frames,
      "num_frames_batch": segment_num_frames,
      "video_segment_ids": video_segment_ids
  }


def inference(reader, train_dir, data_pattern, out_file_location, batch_size,
              top_k):
  """Inference function."""
  with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True)) as sess, gfile.Open(out_file_location,
                                                      "w+") as out_file:
    video_id_batch, video_batch, num_frames_batch = get_input_data_tensors(
        reader, data_pattern, batch_size)
    inference_model_name = "segment_inference_model" if FLAGS.segment_labels else "inference_model"
    checkpoint_file = os.path.join(train_dir, "inference_model",
                                   inference_model_name)
    if not gfile.Exists(checkpoint_file + ".meta"):
      raise IOError("Cannot find %s. Did you run eval.py?" % checkpoint_file)
    meta_graph_location = checkpoint_file + ".meta"
    logging.info("loading meta-graph: " + meta_graph_location)

    if FLAGS.output_model_tgz:
      with tarfile.open(FLAGS.output_model_tgz, "w:gz") as tar:
        for model_file in glob.glob(checkpoint_file + ".*"):
          tar.add(model_file, arcname=os.path.basename(model_file))
        tar.add(os.path.join(train_dir, "model_flags.json"),
                arcname="model_flags.json")
      print("Tarred model onto " + FLAGS.output_model_tgz)
    with tf.device("/cpu:0"):
      saver = tf.train.import_meta_graph(meta_graph_location,
                                         clear_devices=True)
    logging.info("restoring variables from " + checkpoint_file)
    saver.restore(sess, checkpoint_file)
    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(
        set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    whitelisted_cls_mask = None
    if FLAGS.segment_labels:
      final_out_file = out_file
      out_file = tempfile.NamedTemporaryFile()
      logging.info(
          "Segment temp prediction output will be written to temp file: %s",
          out_file.name)
      if FLAGS.segment_label_ids_file:
        whitelisted_cls_mask = np.zeros((predictions_tensor.get_shape()[-1],),
                                        dtype=np.float32)
        segment_label_ids_file = FLAGS.segment_label_ids_file
        if segment_label_ids_file.startswith("http"):
          logging.info("Retrieving segment ID whitelist files from %s...",
                       segment_label_ids_file)
          segment_label_ids_file, _ = urllib.request.urlretrieve(
              segment_label_ids_file)
        with tf.io.gfile.GFile(segment_label_ids_file) as fobj:
          for line in fobj:
            try:
              cls_id = int(line)
              whitelisted_cls_mask[cls_id] = 1.
            except ValueError:
              # Simply skip the non-integer line.
              continue

    out_file.write(u"VideoId,LabelConfidencePairs\n".encode("utf8"))

    try:
      while not coord.should_stop():
        video_id_batch_val, video_batch_val, num_frames_batch_val = sess.run(
            [video_id_batch, video_batch, num_frames_batch])
        if FLAGS.segment_labels:
          results = get_segments(video_batch_val, num_frames_batch_val, 5)
          video_segment_ids = results["video_segment_ids"]
          video_id_batch_val = video_id_batch_val[video_segment_ids[:, 0]]
          video_id_batch_val = np.array([
              "%s:%d" % (x.decode("utf8"), y)
              for x, y in zip(video_id_batch_val, video_segment_ids[:, 1])
          ])
          video_batch_val = results["video_batch"]
          num_frames_batch_val = results["num_frames_batch"]
          if input_tensor.get_shape()[1] != video_batch_val.shape[1]:
            raise ValueError("max_frames mismatch. Please re-run the eval.py "
                             "with correct segment_labels settings.")

        predictions_val, = sess.run([predictions_tensor],
                                    feed_dict={
                                        input_tensor: video_batch_val,
                                        num_frames_tensor: num_frames_batch_val
                                    })
        now = time.time()
        num_examples_processed += len(video_batch_val)
        elapsed_time = now - start_time
        logging.info("num examples processed: " + str(num_examples_processed) +
                     " elapsed seconds: " + "{0:.2f}".format(elapsed_time) +
                     " examples/sec: %.2f" %
                     (num_examples_processed / elapsed_time))
        for line in format_lines(video_id_batch_val, predictions_val, top_k,
                                 whitelisted_cls_mask):
          out_file.write(line)
        out_file.flush()

    except tf.errors.OutOfRangeError:
      logging.info("Done with inference. The output file was written to " +
                   out_file.name)
    finally:
      coord.request_stop()

      if FLAGS.segment_labels:
        # Re-read the file and do heap sort.
        # Create multiple heaps.
        logging.info("Post-processing segment predictions...")
        heaps = {}
        out_file.seek(0, 0)
        for line in out_file:
          segment_id, preds = line.decode("utf8").split(",")
          if segment_id == "VideoId":
            # Skip the headline.
            continue
          preds = preds.split(" ")
          pred_cls_ids = [int(preds[idx]) for idx in range(0, len(preds), 2)]
          pred_cls_scores = [
              float(preds[idx]) for idx in range(1, len(preds), 2)
          ]
          for cls, score in zip(pred_cls_ids, pred_cls_scores):
            if not whitelisted_cls_mask[cls]:
              # Skip non-whitelisted classes.
              continue
            if cls not in heaps:
              heaps[cls] = []
            if len(heaps[cls]) >= FLAGS.segment_max_pred:
              heapq.heappushpop(heaps[cls], (score, segment_id))
            else:
              heapq.heappush(heaps[cls], (score, segment_id))
        logging.info("Writing sorted segment predictions to: %s",
                     final_out_file.name)
        final_out_file.write("Class,Segments\n")
        for cls, cls_heap in heaps.items():
          cls_heap.sort(key=lambda x: x[0], reverse=True)
          final_out_file.write("%d,%s\n" %
                               (cls, " ".join([x[1] for x in cls_heap])))
        final_out_file.close()

      out_file.close()

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  if FLAGS.input_model_tgz:
    if FLAGS.train_dir:
      raise ValueError("You cannot supply --train_dir if supplying "
                       "--input_model_tgz")
    # Untar.
    if not os.path.exists(FLAGS.untar_model_dir):
      os.makedirs(FLAGS.untar_model_dir)
    tarfile.open(FLAGS.input_model_tgz).extractall(FLAGS.untar_model_dir)
    FLAGS.train_dir = FLAGS.untar_model_dir

  flags_dict_file = os.path.join(FLAGS.train_dir, "model_flags.json")
  if not file_io.file_exists(flags_dict_file):
    raise IOError("Cannot find %s. Did you run eval.py?" % flags_dict_file)
  flags_dict = json.loads(file_io.FileIO(flags_dict_file, "r").read())

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      flags_dict["feature_names"], flags_dict["feature_sizes"])

  if flags_dict["frame_features"]:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if not FLAGS.output_file:
    raise ValueError("'output_file' was not specified. "
                     "Unable to continue with inference.")

  if not FLAGS.input_data_pattern:
    raise ValueError("'input_data_pattern' was not specified. "
                     "Unable to continue with inference.")

  inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
            FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
