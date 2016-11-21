"""Calculate the mean average precision.

It provides an interface for calculating interpolated mean average precision
for an entire list or the top-n ranked items.

Example usages:
We first call the function accumulate many times to process parts of the ranked
list. After processing all the parts, we call peek_interpolated_map_at_n
to calculate the mean average precision.

```
import random

p = np.array([[random.random() for _ in xrange(50)] for _ in xrange(1000)])
a = np.array([[random.choice([0, 1]) for _ in xrange(50)]
     for _ in xrange(1000)])

# interpolated mean average precision at 1000 for 50 classes.
calculator = mean_average_precision_calculator.MeanAveragePrecisionCaculator(
            num_class=50, top_n_array=[1000 for _ in xrange(50)])
calculator.accumulate(p, a)
aps = calculator.peek_interpolated_map_at_n()
```
"""

import numpy
import average_precision_calculator


class MeanAveragePrecisionCaculator(object):
  """This class is to calculate mean average precision.
  """

  def __init__(self, num_class, top_n_array=None):
    """Construct a calculator to calculate the (macro) average precision.

    Args:
      num_class: A positive Integer specifying the number of classes.
      top_n_array: A list of positive integers specifying the top n for each
      class. The top n in each class will be used to calculate its average
      precision at n.
      The size of the array must be num_class.

    Raises:
      ValueError: An error occurred when num_class is not a positive integer;
      or the top_n_array is not a list of positive integers.
    """
    if not isinstance(num_class, int) or num_class <= 1:
      raise ValueError("num_class must be a positive integer.")
    if top_n_array is None:
      top_n_array = [1000 for _ in xrange(num_class)]
    if len(top_n_array) != num_class:
      raise ValueError("top_n_array must have the length of " + str(num_class))

    self._ap_calculators = []  # member of AveragePrecisionCalculator
    self._num_class = num_class  # total number of classes
    # top_n array to calculate ap@n for each class
    self._top_n_array = top_n_array
    for top_n in top_n_array:
      self._ap_calculators.append(
          average_precision_calculator.AveragePrecisionCalculator(top_n))

  def _check_input(self, predictions, actuals):
    if (not isinstance(predictions, numpy.ndarray) or
        not isinstance(actuals, numpy.ndarray) or
        predictions.shape != actuals.shape or len(predictions.shape) != 2):
      return False
    else:
      return True

  def accumulate(self, predictions, actuals):
    """Accumulate the predictions and their ground truth labels.

    Args:
      predictions: a numpy 2-D array storing the prediction scores. Each
      column represents a class and each row indicates a sample.
      actuals: a numpy 2-D array storing the ground truth labels. Each
      column represents a class and each row indicates a sample. Any value
      larger than 0 will be treated as positives, otherwise as negatives.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
      does not match.
    """
    if not self._check_input(predictions, actuals):
      raise ValueError("predictions and actuals must be the numpy 2-D"
                       "array of the same shape.")
    calculators = self._ap_calculators
    for i in xrange(predictions.shape[1]):
      calculators[i].accumulate(predictions[:, i], actuals[:, i])

  def clear(self):
    for calculator in self._ap_calculators:
      calculator.clear()

  def is_empty(self):
    return ([calculator.heap_size for calculator in self._ap_calculators] ==
            [0 for _ in range(self._num_class)])

  def peek_interpolated_map_at_n(self, inter_points=1000):
    """Peek the interpolated average precision at n.

    Args:
      inter_points: the interpolating points for calculating the
      mean average precision (default 1000).

    Returns:
      An array of non-interpolated average precision at n (default 0) for each
      class.
    """
    aps = [self._ap_calculators[i].peek_interpolated_ap_at_n(inter_points)
           for i in xrange(self._num_class)]
    return aps
