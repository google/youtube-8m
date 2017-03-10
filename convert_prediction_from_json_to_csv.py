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

"""Utility to convert the output of batch prediction into a CSV submission.

It converts the JSON files created by the command
'gcloud beta ml jobs submit prediction' into a CSV file ready for submission.
"""

import json
import tensorflow as tf

from builtins import range
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging


FLAGS = flags.FLAGS

if __name__ == '__main__':

  flags.DEFINE_string(
      "json_prediction_files_pattern", None,
      "Pattern specifying the list of JSON files that the command "
      "'gcloud beta ml jobs submit prediction' outputs. These files are "
      "located in the output path of the prediction command and are prefixed "
      "with 'prediction.results'.")
  flags.DEFINE_string(
      "csv_output_file", None,
      "The file to save the predictions converted to the CSV format.")


def get_csv_header():
  return "VideoId,LabelConfidencePairs\n"

def to_csv_row(json_data):

  video_id = json_data["video_id"]

  class_indexes = json_data["class_indexes"]
  predictions = json_data["predictions"]

  if isinstance(video_id, list):
    video_id = video_id[0]
    class_indexes = class_indexes[0]
    predictions = predictions[0]

  if len(class_indexes) != len(predictions):
    raise ValueError(
        "The number of indexes (%s) and predictions (%s) must be equal." 
        % (len(class_indexes), len(predictions)))

  return (video_id.decode('utf-8') + "," + " ".join("%i %f" % 
      (class_indexes[i], predictions[i]) 
      for i in range(len(class_indexes))) + "\n")

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.json_prediction_files_pattern:
    raise ValueError(
        "The flag --json_prediction_files_pattern must be specified.")

  if not FLAGS.csv_output_file:
    raise ValueError("The flag --csv_output_file must be specified.")

  logging.info("Looking for prediction files with pattern: %s", 
               FLAGS.json_prediction_files_pattern)

  file_paths = gfile.Glob(FLAGS.json_prediction_files_pattern)  
  logging.info("Found files: %s", file_paths)

  logging.info("Writing submission file to: %s", FLAGS.csv_output_file)
  with gfile.Open(FLAGS.csv_output_file, "w+") as output_file:
    output_file.write(get_csv_header())

    for file_path in file_paths:
      logging.info("processing file: %s", file_path)

      with gfile.Open(file_path) as input_file:

        for line in input_file: 
          json_data = json.loads(line)
          output_file.write(to_csv_row(json_data))

    output_file.flush()
  logging.info("done")

if __name__ == "__main__":
  app.run()
