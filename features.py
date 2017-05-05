#!/usr/bin/env python3
import argparse
import logging
import os
import pickle

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from features_utils import write_to_tfrecord

from t1000.embedding import video

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])


def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    '''Dequantize the feature from the byte format to the float format.
    Args:
        feat_vector: the input 1-d vector.
        max_quantized_value: the maximum of the quantized value.
        min_quantized_value: the minimum of the quantized value.
    Returns:
        A float vector which has the same shape as feat_vector.
    '''
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def quantize(features, max_quantized_value=2, min_quantized_value=-2):
    '''Quantize the feature from the float format to the byte format.
    Args:
        features: the input 1-d vector.
        max_quantized_value: the maximum of the quantized value.
        min_quantized_value: the minimum of the quantized value.
    Returns:
        A float vector which has the same shape as feat_vector.
    '''
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    bias = (quantized_range / 512.0) + min_quantized_value
    features = (features - bias) / quantized_range * 255
    return np.uint8(features)


def pca(features, pca_model_path):
    '''
    Reduces the dimensionality of data
    '''

    logger = logging.getLogger(__name__)
    logger.debug("Performing PCA")

    # open PCA model
    with open(pca_model_path, "rb") as file:
        pca = pickle.load(file, encoding='latin1')

    # reduce dimension and quantize data
    features = pca.transform(features)
    features = np.array(features, dtype=np.float32)

    return features


def incepction_v3(frames, model_dir):
    '''
    Extract incepction_v3 features from list of video frames.
    Inputs:
    '''
    logger = logging.getLogger(__name__)
    logger.debug("Extracting inception features")

    # load incepction 3 graph
    with gfile.FastGFile(model_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    img_features = []
    with tf.Session() as sess:
        for frame in frames:
            # get tensor from network
            pool3_layer = sess.graph.get_tensor_by_name('pool_3:0')
            predictions = sess.run(pool3_layer, {'DecodeJpeg:0': frame})
            # concatenate features
            features = np.squeeze(predictions)
            img_features.append(features)

    return np.array(img_features, dtype=np.float32)


def feature_pipeline(
    video_path,
    inception_model_path,
    pca_model_path,
    quantize=False):
    '''
    Feature pipeline similat to youtube-8m whitepaper
    '''

    # step-by-step feature extraction pipeline
    frames = video.extract_frames(video_path)
    features = incepction_v3(frames, inception_model_path)
    features = pca(features, pca_model_path)
    featues = quantize(features) if quantize else features

    # take average in columns
    features = np.mean(features, axis=0)

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--video_path", help="Path to transformed video",
        required=True)

    parser.add_argument(
        "-i","--inception3-path", help="Path to inception model",
        required=True)

    parser.add_argument(
        "-p", "--pca-path", help="Path pca model path",
        required=True)

    parser.add_argument(
        "--output_path", help="path to tfrecord_features",
        default="/data/video/video-level-features")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("Extracting features from: {0}".format(args.video_path))

    features = feature_pipeline(args.video_path, args.inception3_path, args.pca_path)

    logger.info("Saving features as TFRecordfile")

    output_file = os.path.join(args.output_path, "test.tfrecord")
    write_to_tfrecord(args.video_path, [], features, output_file)
