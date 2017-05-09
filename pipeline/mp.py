import logging
import multiprocessing
import os
import pickle

import numpy as np
import tensorflow as tf

from t1000.embedding import video
from tensorflow.python.platform import gfile

def chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


class Producer(multiprocessing.Process):
    def __init__(self, items, idx, queue, logging_step=100):
        super(Producer, self).__init__()
        self.items = items
        self.queue = queue
        self.idx = idx
        self.logging_step = logging_step

    def run(self):
        logger = logging.getLogger(__name__)
        logger.info("[Producer %d] Starting" % self.idx)
        processed_items = 0
        while self.items:
            # get items
            video_id, video_tags, video_path = self.items.pop()

            if processed_items % self.logging_step == 0:
                logger.info("[Producer %d] Downloading URL for item number %d" % (
                    self.idx, processed_items))

            try:
                frames = video.extract_frames(video_path)
                self.queue.put((video_id, frames, video_tags))

            except Exception as e:
                logger.exception("[Producer %d] Exception happened while processing video %s for item number %d" %(
                    self.idx, video_id, processed_items))

            if processed_items % self.logging_step == 0:
                logger.info("[Producer %d] Extracted frames from %s for item number %d" % (
                    self.idx, video_id, processed_items))

            processed_items = processed_items + 1

        logger.info("[Producer %d] Signaling end of work" % self.idx)
        self.queue.put(None)

        logger.info("[Producer %d] Finished" % self.idx)
        return

class Consumer(multiprocessing.Process):
    def __init__(self, queues, model_dir, data_dir, logging_step = 100):
        super(Consumer, self).__init__()
        self.queues = queues
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logging_step = logging_step

    def run(self):
        logger = logging.getLogger(__name__)
        logger.info("Starting consumer")

            # load incepction 3 graph
        with gfile.FastGFile(self.model_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        logger.info("Loaded graph")

        with tf.Session() as sess:
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


        logger.info("Ending consumer")
        return

def fetch(work, nprod, model_path, data_path, logging_step = 100):
    work = chunks(work, nprod)
    #make reader for reading data. lets call this object Producer
    producers = []
    queues = []
    for idx in range(nprod):
        queues.append(multiprocessing.Queue())
        producers.append(Producer(work[idx], idx, queues[idx], logging_step))

    #make receivers for the data. Lets call these Consumers
    #Each consumer is assigned a queue
    consumer_object = Consumer(queues, model_path, data_path, logging_step)
    consumer_object.start()

    # start the producer processes
    for producer_object in producers:
        producer_object.start()


    #Join all started processes
    consumer_object.join()

    for producer_object in producers:
        producer_object.join()
