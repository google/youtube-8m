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
"""Tests for feature_extractor."""

import pickle
import json
import os
import feature_extractor
import numpy
from PIL import Image
from tensorflow.python.platform import googletest


def _FilePath(filename):
  return os.path.join('testdata', filename)


def _MeanElementWiseDifference(a, b):
  """Calculates element-wise percent difference between two numpy matrices."""
  difference = numpy.abs(a - b)
  denominator = numpy.maximum(numpy.abs(a), numpy.abs(b))

  # We dont care if one is 0 and another is 0.01
  return (difference / (0.01 + denominator)).mean()


class FeatureExtractorTest(googletest.TestCase):

  def setUp(self):
    self._extractor = feature_extractor.YouTube8MFeatureExtractor()

  def testPCAOnFeatureVector(self):
    sports_1m_test_data = cPickle.load(open(_FilePath('sports1m_frame.pkl')))
    actual_pca = self._extractor.apply_pca(sports_1m_test_data['original'])
    expected_pca = sports_1m_test_data['pca']
    self.assertLess(_MeanElementWiseDifference(actual_pca, expected_pca), 1e-5)


if __name__ == '__main__':
  googletest.main()
