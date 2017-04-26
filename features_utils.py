import tensorflow as tf

import hashlib

def cheap_hash(txt, length=11):
    '''
    Hashes a sting
    '''
    hash = hashlib.sha1()
    hash.update(txt)
    return hash.hexdigest()[:length]


def write_to_tfrecord(video_id, labels, features, output_file):
    '''
    Serializes featues as tf.train.Example protobuff object and stores it
    TFRecord file.

    Args:
        video_id - video filename or id
        labels   - list of tags, can be an empty list
        features - 1-D vector with video features
    '''
    writer = tf.python_io.TFRecordWriter(output_file)
    example = tf.train.Example(features=tf.train.Features(feature={
        'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[cheap_hash(video_idi)])),
        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
        'mean_rgb': tf.train.Feature(float_list=tf.train.FloatList(value=features))
        }
    ))

    writer.write(example.SerializeToString())
    writer.close()


def read_from_tfrecord(filenames):
     tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
     reader = tf.TFRecordReader()
     _, tfrecord_serialized = reader.read(tfrecord_file_queue)

     tfrecord_features = tf.parse_single_example(
         tfrecord_serialized,
         features={
             'video_id': tf.FixedLenFeature([], tf.string),
             'labels': tf.VarLenFeature(tf.int64),
             'mean_rgb': tf.FixedLenFeature([], tf.float32),
         },name='features')
