"""Provides functions to help with evaluating models."""
import datetime
import numpy

from tensorflow.python.platform import gfile

import mean_average_precision_calculator as map_calculator


def calculate_hit_at_one(predictions, actuals):
  """Performs a local (numpy) calculation of the hit at one.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average hit at one across the entire batch.
  """
  top_prediction = numpy.argmax(predictions, 1)
  hits = actuals[numpy.arange(actuals.shape[0]), top_prediction]
  return numpy.average(hits)


def calculate_precision_at_equal_recall_rate(predictions, actuals):
  """Performs a local (numpy) calculation of the PERR.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average precision at equal recall rate across the entire batch.
  """
  aggregated_precision = 0.0
  num_videos = actuals.shape[0]
  for row in numpy.arange(num_videos):
    num_labels = int(numpy.sum(actuals[row]))
    top_indices = numpy.argpartition(predictions[row],
                                     -num_labels)[-num_labels:]
    item_precision = 0.0
    for label_index in top_indices:
      if predictions[row][label_index] > 0:
        item_precision += actuals[row][label_index]
    item_precision /= top_indices.size
    aggregated_precision += item_precision
  aggregated_precision /= num_videos
  return aggregated_precision


def topk_thresholding_matrix(predictions, k=20):
  """Get the top_k for each prediction in the predictions matrix.

  Args:
    predictions: A numpy matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    k: the top k non-zero entries to preserve in each prediction.

  Returns:
    The predictions matrix after applying the topk thresholding.

  Raises:
    ValueError: An error occurred when the k is a positive integer.
  """
  if k <= 0:
    raise ValueError("k must be a positive integer.")
  k = min(k, predictions.shape[1])
  for row in range(predictions.shape[0]):
    predictions[row] = _topk_thresholding_array(predictions[row], k)
  return predictions

def _topk_thresholding_array(arr, k=20):
  """Get the top_k for a 1-d numpy array."""
  m = len(arr)
  k = min(k, m)
  out_vector = numpy.copy(arr)
  idx = numpy.argpartition(arr, m - k)[0:m - k]
  out_vector[idx] = 0
  return out_vector

def _quantize(predictions, num_of_bins=1e03):
  """Quantize the predictions.

  Args:
    predictions: A numpy matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    num_of_bins: a positive number of bins used in quantization.

  Returns:
    The quantized predictions matrix.

  Raises:
    ValueError: An error occurred when the num_of_bins is not positive.
  """
  if num_of_bins <= 0:
    raise ValueError("num_of_bins must be positive.")
  return numpy.floor(predictions * num_of_bins) / float(num_of_bins)

class EvaluationMetrics(object):
  """A class to store the evaluation metrics."""

  def __init__(self, num_class, top_n_array=None):
    """Construct an EvaluationMetrics object to store the evaluation metrics.

    Args:
      num_class: A positive integer specifying the number of classes.
      top_n_array: A list of positive integers specifying the top n for
        calculating the average precision for each class.

    Raises:
      ValueError: An error occurred when MeanAveragePrecisionCaculator cannot
        not be constructed.
    """
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator = map_calculator.MeanAveragePrecisionCaculator(
        num_class, top_n_array)

  def accumulate(self, predictions, labels, loss,
                 accumulate_average_precision=False):
    """Accumulate the metrics calculated locally for this mini-batch.

    Args:
      predictions: A numpy matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      labels: A numpy matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.
      loss: A numpy array containing the loss for each sample.
      accumulate_average_precision: whether to accumulate average precision.

    Returns:
      dictionary: a dictionary storing the metrics for the mini-batch.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
        does not match.
    """
    mean_hit_at_one = calculate_hit_at_one(predictions, labels)
    mean_perr = calculate_precision_at_equal_recall_rate(predictions, labels)
    mean_loss = numpy.mean(loss)

    if accumulate_average_precision:
      # Take the top 20 predictions and use 1000 bins.
      predictions_val = _quantize(predictions, num_of_bins=10**3)
      predictions_val = topk_thresholding_matrix(predictions_val, k=20)
      self.map_calculator.accumulate(predictions_val, labels)

    self._num_examples += 1
    self.sum_hit_at_one += mean_hit_at_one * labels.shape[0]
    self.sum_perr += mean_perr * labels.shape[0]
    self.sum_loss += mean_loss * labels.shape[0]

    return {"hit_at_one": mean_hit_at_one, "perr": mean_perr, "loss": mean_loss}

  def get(self, num_accumulated_examples):
    """Calculate the evaluation metrics for the whole epoch.

    Raises:
      ValueError: If no examples were accumulated.

    Returns:
      dictionary: a dictionary storing the evaluation metrics for the epoch. The
        dictionary has the fields: avg_hit_at_one, avg_perr, avg_loss, and
        aps (default nan).
    """
    if self._num_examples <= 0:
      raise ValueError("total_sample must be positive.")
    avg_hit_at_one = self.sum_hit_at_one / self._num_examples
    avg_perr = self.sum_perr / self._num_examples
    avg_loss = self.sum_loss / self._num_examples

    aps = numpy.core.numeric.nan
    if not self.map_calculator.is_empty():
      aps = self.map_calculator.peek_interpolated_map_at_n(inter_points=1000)

    epoch_info_dict = {}
    return {"avg_hit_at_one": avg_hit_at_one, "avg_perr": avg_perr,
            "avg_loss": avg_loss, "aps": aps}

  def clear(self):
    """Clear the evaluation metrics and reset the EvaluationMetrics object."""
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator.clear_map()
    self._num_examples = 0
