import copy
import pickle
import logging
import os

import numpy as np
import tensorflow as tf

from t1000.embedding import video
from tensorflow.python.platform import gfile

def frame_iterator(work, logging_step = 20):
    logger = logging.getLogger(__name__)
    for index, (video_id, video_tags, video_path) in enumerate(work):

        if index % logging_step == 0:
            logger.info("Downloading URL for item number %d" % index)

        try:
            frames = video.extract_frames(video_path)

            yield video_id, frames, video_tags

        except ValueError as e:
            logger.exception("Exception happened while processing video %s for item number %d" %(
                video_id, index))
            pass

def extract_incepction_v3(frame_iterator, model_dir, data_dir, logging_step = 100):
    '''
    Extract inception_v3 features from frame generator.

    Inputs:
    frame_iterator - an iterator yielding video frames
    model_dir      - a directory to inception model
    logging_step   - log progress after this number of steps
    '''
    logger = logging.getLogger(__name__)
    logger.info("Extracting inception features")

    # load inception 3 graph
    with gfile.FastGFile(model_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


    with tf.Session() as sess:
        # TODO: add queuing and batching for optimal performance
        for index, item in enumerate(frame_iterator):
            item_id, frames, tags = item

            if index % logging_step == 0:
                logger.info("Extracting features from video %s [%d]" % (item_id, index))


            img_features = []
            for frame in frames:
                # get tensor from network
                pool3_layer = sess.graph.get_tensor_by_name('pool_3:0')
                predictions = sess.run(pool3_layer, {'DecodeJpeg:0': frame})

                # concatenate features
                features = np.squeeze(predictions)
                img_features.append(features)

            file_name = os.path.join(data_dir, '{0}.pickle'.format(item_id))
            fv3_features = np.array(img_features, dtype=np.float32)

            with open(file_name, 'wb') as handle:
                pickle.dump((fv3_features, tags), handle, protocol=pickle.HIGHEST_PROTOCOL)

            if index % logging_step == 0:
                logger.info("Extracting features from video %s [%d] [DONE!]" % (item_id, index))

def fetch(work, model_path, data_path, logging_step = 100):
    extract_incepction_v3(
        frame_iterator(work),
        model_path, data_path, logging_step)

