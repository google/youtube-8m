#!/usr/bin/env python3
import argparse
import cv2
import logging
import numpy as np
import os
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile
import yaml

os.environ['GLOG_minloglevel'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_file(file_name):
    video_capture = cv2.VideoCapture(file_name)
    return video_capture

def get_scene_from_frames(video_capture, frames_directory_path):        
    scenes = []
    # Save one frame per interval
    i = 0
    while video_capture.isOpened():
        position = 1000 * i
        video_capture.set(cv2.CAP_PROP_POS_MSEC, position) # Go to the k sec. position
        success, image = video_capture.read()
        
        if not success:
            break
        frame_path = os.path.join(
                frames_directory_path, '%d.png') % i
        scenes.append(frame_path)
        cv2.imwrite(frame_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        i += 1
        if i >= 360:
            break

    video_capture.release()
    return scenes

def create_graph(model_dir):
    with gfile.FastGFile(model_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def compute_features(image, sess, next_to_last_tensor):
    """
    Extract features from image using inception v3 cnn.
    Inputs:
    -image: path to image
    -sess: tensorflow session
    -next_to_last_tensor: tensorflow tensor
    Returns:
    -features: list of extracted features
    """
    if not gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = gfile.FastGFile(image, 'rb').read()
    predictions = sess.run(next_to_last_tensor, {
                           'DecodeJpeg/contents:0': image_data})
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

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for image_path in list_images_paths:
            features = compute_features(image_path, sess, next_to_last_tensor)
            os.remove(image_path)
            img_features.append(features)

    return np.array(img_features, dtype = np.float32)
        

def process_scenes(scenes):
    def convert_to_jpeg(list_img_filenames):
        for img_path in list_img_filenames:
            img_png = Image.open(img_path)
            img_png.save(img_path, 'jpeg')  # convert to jpeg
            
    try: 
        convert_to_jpeg(scenes)
        if scenes:
            img_features_dict = extract_all_features(scenes)
    except IOError:
            logging.info("No frame found in a video")
            
    return img_features_dict

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
    video_capture = load_file(video_name)
    scenes = get_scene_from_frames(video_capture, '.')
    return scenes

def scenes_features(scenes, inception_model_path):
    create_graph(inception_model_path)
    features = process_scenes(scenes)
    return features

def decrease_dimension(features, pca_model_path):
    # open PCA model
    with open(pca_model_path, "rb") as file:
        pca_obj = pickle.load(file)
    
    # reduce dimension and quantize data
    features = pca_obj.transform(features)
    features = np.array(features, dtype=np.float32)
    q_features = quantize(features)
    return q_features

def video_features(video_name, inception_model_path, pca_model_path):
    scenes = video_scenes(video_name)
    features = scenes_features(scenes, inception_model_path)
    
    q_features = decrease_dimension(features, pca_model_path)
    
    # take average in columns
    final_vector = np.mean(q_features, axis = 0)
    final_vector = final_vector.reshape(1, len(final_vector))
    return final_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", help="Path to transformed video", required=True)
    parser.add_argument("-m", "--model", help="Path inception model", required=True)
    parser.add_argument("-p", "--pca", help="Path pca model", required=True)
    args = parser.parse_args()
    
    video_name = args.video_path
    inception_model_path = args.model
    pca_model_path = args.pca
    video_features_vector = video_features(video_name, inception_model_path, pca_model_path)