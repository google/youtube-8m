#!/usr/bin/env python3
import argparse
import logging
import os
import pickle

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from features_utils import write_to_tfrecord

from t1000.embedding import video

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


def get_scene_from_frames(video_capture, frames_directory_path):
    #TODO: use existing code to process video with ffmpeg
    scenes = []
    # Save one frame per interval
    i = 0
    while video_capture.isOpened():
        position = 1000 * i
        video_capture.set(
            cv2.CAP_PROP_POS_MSEC,
            position)  # Go to the k sec. position
        success, image = video_capture.read()

        if not success:
            break

        frame_path = os.path.join(frames_directory_path, '%d.jpeg') % i
        scenes.append(frame_path)

        cv2.imwrite(frame_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        i += 1
        if i >= 360:
            break

    video_capture.release()
    return scenes


def compute_features(image, sess):
    """
    Extract features from image using inception v3 cnn.
    Inputs:
    -image: path to image
    -sess: tensorflow session
    -next_to_last_tensor: tensorflow tensor
    Returns:
    -features: list of extracted features
    """
    logger = logging.getLogger(__name__)

    if not gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)

    image_data = gfile.FastGFile(image, 'rb').read()

    # What is this?
    logger.info(type(image_data))

    # get tensor from network
    pool3_layer = sess.graph.get_tensor_by_name('pool_3:0')

    predictions = sess.run(pool3_layer, {'DecodeJpeg/contents:0': image_data})
    features = np.squeeze(predictions)

    return features


def extract_all_features(list_images_paths):
    """
    Extract features from list of images.
    Inputs:
    -image_list: list of image paths
    Returns:
    -img_features: list of extracted features
    """
    img_features = []

    with tf.Session() as sess:


        for image_path in list_images_paths:
            features = compute_features(image_path, sess, next_to_last_tensor)
            os.remove(image_path)
            img_features.append(features)

    return np.array(img_features, dtype=np.float32)

def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.
    Args:
        feat_vector: the input 1-d vector.
        max_quantized_value: the maximum of the quantized value.
        min_quantized_value: the minimum of the quantized value.
    Returns:
        A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def quantize(features, max_quantized_value=2, min_quantized_value=-2):
    """Quantize the feature from the float format to the byte format.
    Args:
        features: the input 1-d vector.
        max_quantized_value: the maximum of the quantized value.
        min_quantized_value: the minimum of the quantized value.
    Returns:
        A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    bias = (quantized_range / 512.0) + min_quantized_value
    features = (features - bias) / quantized_range * 255
    return np.uint8(features)


def video_scenes(video_name):
    video_capture = cv2.VideoCapture(file_name)
    scenes = get_scene_from_frames(video_capture, '.')
    return scenes


def scenes_features(scenes, model_dir):
    with gfile.FastGFile(model_dir, 'rb') as f
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    try:
        if scenes:
            features = extract_all_features(scenes)
    except IOError:
        logging.info("No frame found in a video")

    return features


def decrease_dimension(features, pca_model_path, quantize=False):
    '''
    Reduces the dimensionality of data
    '''

    # open PCA model
    with open(pca_model_path, "rb") as file:
        pca = pickle.load(file)

    # reduce dimension and quantize data
    features = pca.transform(features)
    features = np.array(features, dtype=np.float32)

    return quantize(features) if quantize else features


def video_features(video_name, inception_model_path, pca_model_path):
    scenes = video_scenes(video_name)
    features = scenes_features(scenes, inception_model_path)

    q_features = decrease_dimension(features, pca_model_path)

    # take average in columns
    final_vector = np.mean(q_features, axis=0)

    return final_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--video_path",
        help="Path to transformed video",
        required=True)

    parser.add_argument(
        "-m","--model",
        help="Path inception model",
        required=True)

    parser.add_argument(
        "-p", "--pca",
        help="Path pca model",
        required=True)

    parser.add_argument(
        "--output_path",
        help="path to tfrecord_features",
        default="/data/video/video-level-features")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    logger.info("Extracting features")
    video_features_vector = video_features(args.video_path, args.model, args.pca)

    logger.info(type(video_features_vector))
    logger.info(video_features_vector.shape)
    logger.info("Saving features as TFRecordfile")

    output_file = os.path.join(args.output_path, "test.tfrecord")
    write_to_tfrecord(args.video_path, [], video_features_vector, output_file)
