# Lint as: python3
import numpy as np
import tensorflow as tf
from tensorflow import app
from tensorflow import flags

FLAGS = flags.FLAGS


def main(unused_argv):
  # Get the input tensor names to be replaced.
  tf.reset_default_graph()
  meta_graph_location = FLAGS.checkpoint_file + ".meta"
  tf.train.import_meta_graph(meta_graph_location, clear_devices=True)

  input_tensor_name = tf.get_collection("input_batch_raw")[0].name
  num_frames_tensor_name = tf.get_collection("num_frames")[0].name

  # Create output graph.
  saver = tf.train.Saver()
  tf.reset_default_graph()

  input_feature_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 1152))
  num_frames_placeholder = tf.placeholder(tf.int32, shape=(None, 1))

  saver = tf.train.import_meta_graph(
      meta_graph_location,
      input_map={
          input_tensor_name: input_feature_placeholder,
          num_frames_tensor_name: tf.squeeze(num_frames_placeholder, axis=1)
      },
      clear_devices=True)
  predictions_tensor = tf.get_collection("predictions")[0]

  with tf.Session() as sess:
    print("restoring variables from " + FLAGS.checkpoint_file)
    saver.restore(sess, FLAGS.checkpoint_file)
    tf.saved_model.simple_save(
        sess,
        FLAGS.output_dir,
        inputs={'rgb_and_audio': input_feature_placeholder,
                'num_frames': num_frames_placeholder},
        outputs={'predictions': predictions_tensor})

    # Try running inference.
    predictions = sess.run(
       [predictions_tensor],
       feed_dict={
          input_feature_placeholder: np.zeros((3, 7, 1152), dtype=np.float32),
          num_frames_placeholder: np.array([[7]], dtype=np.int32)})
    print('Test inference:', predictions)

    print('Model saved to ', FLAGS.output_dir)


if __name__ == '__main__':
  flags.DEFINE_string('checkpoint_file', None, 'Path to the checkpoint file.')
  flags.DEFINE_string('output_dir', None, 'SavedModel output directory.')
  app.run(main)
