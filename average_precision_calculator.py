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

"""Calculate or keep track of the interpolated average precision.

It provides an interface for calculating interpolated average precision for an
entire list or the top-n ranked items. For the definition of the
(non-)interpolated average precision:
http://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf

Example usages:
1) Use it as a static function call to directly calculate average precision for
a short ranked list in the memory.

```
import random

p = np.array([random.random() for _ in xrange(10)])
a = np.array([random.choice([0, 1]) for _ in xrange(10)])

# interpolated average precision using 1000 break points
ap = average_precision_calculator.AveragePrecisionCalculator.interpolated_ap(p,
      a, inter_points=1000)
```

2) Use it as an object for long ranked list that cannot be stored in memory or
the case where partial predictions can be observed at a time (Tensorflow
predictions). In this case, we first call the function accumulate many times
to process parts of the ranked list. After processing all the parts, we call
peek_interpolated_ap_at_n.
```
p1 = np.array([random.random() for _ in xrange(5)])
a1 = np.array([random.choice([0, 1]) for _ in xrange(5)])
p2 = np.array([random.random() for _ in xrange(5)])
a2 = np.array([random.choice([0, 1]) for _ in xrange(5)])

# interpolated average precision at 10 using 1000 break points
calculator = average_precision_calculator.AveragePrecisionCalculator(10)
calculator.accumulate(p1, a1)
calculator.accumulate(p2, a2)
ap3 = calculator.peek_interpolated_ap_at_n(inter_points=1000)
```
"""

import heapq
import random
import numbers

import numpy


class AveragePrecisionCalculator(object):
  """Calculate the average precision and average precision at n."""

  def __init__(self, top_n=None):
    """Construct an AveragePrecisionCalculator to calculate average precision.

    This class is used to calculate the average precision for a single label.

    Args:
      top_n: A positive Integer specifying the average precision at n, or
        None to use all provided data points.

    Raises:
      ValueError: An error occurred when the top_n is not a positive integer.
    """
    if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
      raise ValueError("top_n must be a positive integer or None.")

    self._top_n = top_n  # average precision at n
    self._total_positives = 0  # total number of positives have seen
    self._heap = []  # max heap of (prediction, actual)

  @property
  def heap_size(self):
    """Gets the heap size maintained in the class."""
    return len(self._heap)

  @property
  def num_accumulated_positives(self):
    """Gets the number of positive samples that have been accumulated."""
    return self._total_positives

  def accumulate(self, predictions, actuals, num_positives=None):
    """Accumulate the predictions and their ground truth labels.

    After the function call, we may call peek_ap_at_n to actually calculate
    the average precision.
    Note predictions and actuals must have the same shape.

    Args:
      predictions: a list storing the prediction scores.
      actuals: a list storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.
      num_positives = If the 'predictions' and 'actuals' inputs aren't complete,
      then it's possible some true positives were missed in them. In that case,
      you can provide 'num_positives' in order to accurately track recall.

    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if not num_positives is None:
      if not isinstance(num_positives, numbers.Number) or num_positives < 0:
        raise ValueError("'num_positives' was provided but it wan't a nonzero number.")

    if not num_positives is None:
      self._total_positives += num_positives
    else:
      self._total_positives += numpy.size(numpy.where(actuals > 0))
    topk = self._top_n
    heap = self._heap

    for i in xrange(numpy.size(predictions)):
      if topk is None or len(heap) < topk:
        heapq.heappush(heap, (predictions[i], actuals[i]))
      else:
        if predictions[i] > heap[0][0]:  # heap[0] is the smallest
          heapq.heappop(heap)
          heapq.heappush(heap, (predictions[i], actuals[i]))

  def clear(self):
    """Clear the accumulated predictions."""
    self._heap = []
    self._total_positives = 0

  def peek_interpolated_ap_at_n(self, inter_points=1000):
    r"""Peek the interpolated average precision at n.

    The definition of the interpolated average precision is calculated by:
    $ap = \\sum_{j=0}^interp P(\tau_j) * [R(\tau_j) - R(\tau_{j+1})]$,
    where $\tau_j = j/interp$ and $interp$ is the number of interpolating
    points: inter_points. The $P(\tau_j)$ and $R(\tau_j)$ are the precision
    and the recall at the threshold $\tau_j$ for the top n items.
    The difference between the function interpolated_ap_at_n and interpolated_ap
    is that in calculating the interpolated_ap_at_n, we let the number of
    positive equal min(n, #true_positive), so that a perfect ranked list can
    get ap of 1.0.

    Args:
      inter_points: the interpolating points for calculating average precision
      (default 1000).

    Returns:
      The interpolated average precision at n (default 0).
    """
    if self.heap_size <= 0:
      return 0
    predlists = numpy.array(zip(*self._heap))
    ap = self.interpolated_ap(
        predlists[0],
        predlists[1],
        inter_points=inter_points,
        total_num_positives= min(self._total_positives, self._top_n) if self._top_n else self._total_positives)
    return ap

  @staticmethod
  def interpolated_ap(predictions,
                      actuals,
                      inter_points=1000,
                      total_num_positives=None):
    """Calculate the interpolated average precision.

    Calculate the interpolated average precision at multiple points.
    The complexity of the code is O(nlogn), where n is the number of the sample
    in the list.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      The prediction scores will be normalized to 0.0 and 1.0 if they are not
      in the range.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.
      inter_points: the number interpolating points evenly distributed between
      0 and 1.
      total_num_positives : (optionally) you can specify the number of total
      positive
      in the list. If specified, it will be used in calculation.

    Returns:
      The interpolated average precision.

    Raises:
      ValueError: An error occurred when
      1) the format of the input is not the numpy 1-D array;
      2) the shape of predictions and actuals does not match;
      3) the inter_points is not a positive integer.
    """
    if (not isinstance(predictions, numpy.ndarray) or
        not isinstance(actuals, numpy.ndarray)):
      raise ValueError("predictions and actuals must be the numpy 1-D array.")

    if predictions.shape != actuals.shape:
      raise ValueError("the shape of predictions and actuals does not match.")

    if not isinstance(inter_points, int) or inter_points <= 0:
      raise ValueError("inter_points must be a positive integer.")

    if numpy.min(predictions) < 0 or numpy.max(predictions) > 1:
      predictions = AveragePrecisionCalculator._zero_one_normalize(predictions)

    # add a shuffler to avoid overestimating the ap
    predictions, actuals = AveragePrecisionCalculator._shuffle(predictions,
                                                               actuals)
    sortidx = sorted(
        range(len(predictions)),
        key=lambda k: predictions[k],
        reverse=True)

    if total_num_positives is None:
      numpos = numpy.size(numpy.where(actuals > 0))
    else:
      numpos = total_num_positives

    if numpos == 0:
      return 0

    ap = 0.0
    last_recall = 0
    current_recall = 0
    poscount = 0.0
    taus = [float(i) / inter_points for i in range(inter_points, -1, -1)]
    pt = 0  # pointer in the taus array

    # calculate the ap predictions have to be [0,1].
    # make sure the largest prediction score fall in its proper intersection.
    while predictions[sortidx[0]] < taus[pt]:
      pt += 1

    for i in range(len(sortidx)):
      if predictions[sortidx[i]] < taus[pt]:
        current_recall = poscount / numpos
        precision = poscount / i
        ap += precision * (current_recall - last_recall)
        last_recall = current_recall
        while predictions[sortidx[i]] < taus[pt]:  # increase pointer
          pt += 1
      if actuals[sortidx[i]] > 0:
        poscount += 1.0

    # collect the precision and recall in the last intersection.
    current_recall = poscount / numpos
    ap += poscount / len(sortidx) * (current_recall - last_recall)
    return ap

  @staticmethod
  def _shuffle(predictions, actuals):
    random.seed(0)
    suffidx = random.sample(range(len(predictions)), len(predictions))
    predictions = predictions[suffidx]
    actuals = actuals[suffidx]
    return predictions, actuals

  @staticmethod
  def _zero_one_normalize(predictions, epsilon=1e-7):
    """Normalize the predictions to the range between 0.0 and 1.0.

    For some predictions like SVM predictions, we need to normalize them before
    calculate the interpolated average precision. The normalization will not
    change the rank in the original list and thus won't change the average
    precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      epsilon: a small constant to avoid denominator being zero.

    Returns:
      The normalized prediction.
    """
    denominator = numpy.max(predictions) - numpy.min(predictions)
    ret = (predictions - numpy.min(predictions)) / numpy.max(denominator,
                                                             epsilon)
    return ret
