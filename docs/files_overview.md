# Overview of Files

## Training

*   `train.py`: The primary script for training models.
*   `losses.py`: Contains definitions for loss functions.
*   `models.py`: Contains the base class for defining a model.
*   `video_level_models.py`: Contains definitions for models that take
    aggregated features as input.
*   `frame_level_models.py`: Contains definitions for models that take frame-
    level features as input.
*   `model_util.py`: Contains functions that are of general utility for
    implementing models.
*   `export_model.py`: Provides a class to export a model during training for
    later use in batch prediction.
*   `readers.py`: Contains definitions for the Video dataset and Frame dataset
    readers.

## Evaluation

*   `eval.py`: The primary script for evaluating models.
*   `eval_util.py`: Provides a class that calculates all evaluation metrics.
*   `average_precision_calculator.py`: Functions for calculating average
    precision.
*   `mean_average_precision_calculator.py`: Functions for calculating mean
    average precision.
*   `segment_eval_inference.py`: The primary script to evaluate segment models 
    with Kaggle metrics.

## Inference

*   `inference.py`: Generates an output CSV file containing predictions of the
    model over a set of videos. It optionally generates a tarred file of the
    model.

## Misc

*   `README.md`: This documentation.
*   `utils.py`: Common functions.
