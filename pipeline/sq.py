import copy
import pickle
import logging
import os

import numpy as np
import tensorflow as tf

from t1000.embedding import video
from tensorflow.python.platform import gfile

class FramesIterator:
    '''iterator that yields raw frames from database'''

    def __init__(self, videos):
        self.videos = copy.deepcopy(videos)

    def __iter__(self):
        return self

    def __next__(self):
        logger = logging.getLogger(__name__)
        if self.videos:
            logger.debug("Downloading url")
            video_id, video_tags, video_path = self.videos.pop()

            # this could be done in parallel
            frames = video.extract_frames(video_path)
        else:
            raise StopIteration

        return video_id, frames, video_tags


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
        FramesIterator(work),
        model_path, data_path, logging_step)

