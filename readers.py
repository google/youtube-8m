"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils

def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unpack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.pack(shape)

  resized = tf.concat(axis, [
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.pack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ])

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 'feature_name' feature. The float features are assumed
  to be an average of dequantized values.
  """

  def __init__(self,
               num_classes=4800,
               feature_size=1024,
               feature_name="mean_inc3"):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_size: a positive integer for the feature dimension.
      feature_name: the feature name in the tensorflow record.
    """
    self.num_classes = num_classes
    self.feature_size = feature_size
    self.feature_name = feature_name

  def prepare_reader(self, filename_queue,):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "video_id": tf.FixedLenFeature(
                [], tf.string),
            "labels": tf.VarLenFeature(tf.int64),
            self.feature_name: tf.FixedLenFeature(
                [self.feature_size], tf.float32)
        })

    labels = (tf.cast(
        tf.sparse_to_dense(features["labels"].values, (self.num_classes,), 1),
        tf.bool))
    return features["video_id"], features[
        self.feature_name], labels, tf.constant(1)


class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized 'feature_name' feature. The
  quantized features will be mapped back into a range between
  min_quantized_value and max_quantized_value.
  """

  def __init__(self,
               num_classes=4800,
               feature_size=1024,
               feature_name="inc3",
               max_frames=300):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_size: a positive integer for the feature dimension.
      feature_name: the feature name in the tensorflow record.
      max_frames: the maximum number of frames to process.
    """
    self.num_classes = num_classes
    self.feature_size = feature_size
    self.feature_name = feature_name
    self.max_frames = max_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            self.feature_name: tf.FixedLenSequenceFeature(
                [], dtype=tf.string)
        })

    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (self.num_classes,), 1),
        tf.bool))
    decoded_features = tf.reshape(
        tf.cast(
            tf.decode_raw(features[self.feature_name], tf.uint8), tf.float32),
        [-1, self.feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], self.max_frames)

    video_matrix = utils.Dequantize(decoded_features, max_quantized_value,
                                    min_quantized_value)
    # Pad or truncate to 'max_frames' frames.
    video_matrix = resize_axis(video_matrix, 0, self.max_frames)
    return contexts["video_id"], video_matrix, labels, num_frames
