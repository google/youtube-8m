MODEL_PATH = '/models/image/inception/classify_image_graph_def.pb'
DATA_PATH = '/data/video/video-level-features/'
NPROD = 4
LIMIT = 10
MIN_TAGS = 10

VQUERY = "select post_id, url from videos where status='ok'"
TQUERY = "select id, tags from videos where tags is not NULL"
TAGS = "select tag_id, name, path from content_tags"

import copy
import itertools
import logging
import multiprocessing
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from collections import Counter
from t1000.embedding import video
from tensorflow.python.platform import gfile

from db import fetch, inner_join, filter_videos
from utils.logging import setup_logging


## Sequential dataprocessing
class FramesIterator:
    '''iterator that yields raw frames from database'''

    def __init__(self, videos):
        self.videos = copy.deepcopy(videos)

    def __iter__(self):
        return self

    def __next__(self):
        if self.videos:
            logging.debug("Downloading url")
            video_id, video_tags, video_path = self.videos.pop()

            # this could be done in parallel
            frames = video.extract_frames(video_path)
        else:
            raise StopIteration

        return video_id, frames, video_tags


def extract_incepction_v3(frame_iterator, model_dir, data_dir, logging_step = 100):
    '''
    Extract incepction_v3 features from frame generator.

    Inputs:
    frame_iterator - an iterator yielding video frames
    model_dir      - a directory to inception model 
    logging_step   - log progress after this number of steps
    '''
    logger = logging.getLogger(__name__)
    logger.info("Extracting inception features")

    # load incepction 3 graph
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

def fetch_sq(work, model_path = MODEL_PATH, data_path = DATA_PATH):
    frames = FramesIterator(work)
    extract_incepction_v3(frames, model_path, data_path, 1)


def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

class Producer(multiprocessing.Process):
    def __init__(self, items, idx, queue):
        super(Producer, self).__init__()
        self.items = items
        self.queue = queue
        self.idx = idx

    def run(self):
        logger = logging.getLogger(__name__)
        logger.info("Starting %d producer " % (self.idx ))

        while self.items:
            # get items
            video_id, video_tags, video_path = self.items.pop()

            # extract frames
            logger.debug("[Producer] Downloading url")
            frames = video.extract_frames(video_path)
            logger.debug("[Producer] Extracted frames from %s" % video_id)

            # add items to queue
            self.queue.put((video_id, frames, video_tags))

        logger.info("[Producer] This is it! [%d]" % self.idx)
        self.queue.put(None)

        logger.info('[Producer] Ending producer')
        return

class Consumer(multiprocessing.Process):
    def __init__(self, idx, queues, model_dir, data_dir, logging_step = 100):
        super(Consumer, self).__init__()
        self.queues = queues
        self.idx = idx
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logging_step = logging_step

    def run(self):
        logger = logging.getLogger(__name__)
        logger.info("Starting %d consumer" % (self.idx ))

            # load incepction 3 graph
        with gfile.FastGFile(self.model_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        logger.info("Loaded graph")

        with tf.Session() as sess:
            # TODO: add queuing and batching for optimal performance
            processed_items = 0
            while self.queues:
                for queue in self.queues:
                    item = queue.get()
                    if item is None:
                        self.queues[:] = [q for q in self.queues if q != queue]
                        logger.debug(
                            "[Consumer] Rmoved %s from queues. %d left" % (queue, len(self.queues))
                        )
                        continue


                    item_id, frames, tags = item
                    if processed_items % self.logging_step == 0:
                        logger.info(
                            "[Consumer] Extracting features from video %s [%d]" % (item_id, processed_items)
                        )

                    img_features = []
                    for index, frame in enumerate(frames):
                        # get tensor from network
                        pool3_layer = sess.graph.get_tensor_by_name('pool_3:0')
                        predictions = sess.run(pool3_layer, {'DecodeJpeg:0': frame})

                        # concatenate features
                        features = np.squeeze(predictions)
                        img_features.append(features)


                    file_name = os.path.join(self.data_dir, '{0}.pickle'.format(item_id))
                    fv3_features = np.array(img_features, dtype=np.float32)

                    with open(file_name, 'wb') as handle:
                        pickle.dump((fv3_features, tags), handle, protocol=pickle.HIGHEST_PROTOCOL)

                    if processed_items % self.logging_step == 0:
                        logger.info(
                            "[Consumer] Extracting features from video %s [DONE!][%d]" % (item_id, processed_items))

                    # Increment counter
                    processed_items = processed_items + 1


        logger.info("Ending %d consumer" % (self.idx ))
        return

def fetch_mp(work, nprod = NPROD, model_path = MODEL_PATH, data_path = DATA_PATH):
    work = chunks(work, nprod)
    #make reader for reading data. lets call this object Producer
    producers = []
    queues = []
    for idx in range(nprod):
        queues.append(multiprocessing.Queue())
        producers.append(Producer(work[idx], idx, queues[idx]))

    #make receivers for the data. Lets call these Consumers
    #Each consumer is assigned a queue
    consumer_object = Consumer(1, queues, model_path, data_path)
    consumer_object.start()

    # start the producer processes
    for producer_object in producers:
        producer_object.start()


    #Join all started processes
    consumer_object.join()

    for producer_object in producers:
        producer_object.join()

def run_and_measure(fun, n):
    logger = logging.getLogger(__name__)
    start_time = time.time()
    fun()
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("Elapsed time was %g seconds [%g]" % (elapsed, elapsed/n))


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)

    host='192.95.32.117'

    vdbname='ds-wizards'
    vuser='wizard'
    vpassword='GaG23jVxZhMnQaU53r8o'

    tdbname='ds-content-tags'
    tuser='ds-content-tags'
    tpassword='0fXjWl592vNf1gYvIw8w'

    vres = fetch(host, vdbname, vuser, vpassword, VQUERY)
    vres = [(post_id.split("_")[1], url) for post_id, url in vres]

    tres = fetch(host, tdbname, tuser, tpassword, TQUERY)
    videos = inner_join(tres, vres)

    filtered, t2i, i2t = filter_videos(videos, MIN_TAGS)
    logger.info("Found %d videos with %d unique tags" % (len(filtered), len(t2i)))

    # we will need thid eventually
    tags = {
	tag_id: (name, path) for (tag_id, name, path) in fetch(
            host, tdbname, tuser, tpassword, TAGS)
    }


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    work = filtered[:LIMIT]

    fsq = lambda : fetch_sq(work, model_path=MODEL_PATH, data_path=os.path.join(DATA_PATH, 'seq'))
    fmp = lambda : fetch_mp(work, nprod=4, model_path=MODEL_PATH, data_path=os.path.join(DATA_PATH, 'mp'))

    run_and_measure(fmp, len(work))
