# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
import models

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")


class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}



class DeepMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    hidden1 = slim.fully_connected(
        model_input,
        1024,
        activation_fn=tf.nn.relu,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="hidden1")
    hidden = slim.fully_connected(
        hidden1,
        1024,
        activation_fn=tf.nn.relu,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="hidden")

    gate_activations = slim.fully_connected(
        hidden,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    expert_activations_lvl1 = slim.fully_connected(
        hidden,
        vocab_size * num_mixtures,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts_lvl1")

    expert_activations = slim.fully_connected(
        expert_activations_lvl1,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class DerKorinthenkacker(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     **unused_params):
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
        with tf.name_scope('Feature_XForm'):
            hidden = slim.fully_connected(
                model_input,
                1024,
                activation_fn=tf.nn.relu,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="hidden")

        with tf.name_scope('Gate_Mrudula'):
            gate_activations_mru = slim.fully_connected(
                hidden,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates_mru")

            gating_distribution_mru = tf.nn.softmax(tf.reshape(
                gate_activations_mru,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)

        with tf.name_scope('Expert_Mrudula'):
            expert_activations_lvl1_mru = slim.fully_connected(
                hidden,
                vocab_size * num_mixtures,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts_lvl1_mru")
            expert_activations_mru = slim.fully_connected(
                expert_activations_lvl1_mru,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts_mru")
            expert_distribution_mru = tf.nn.sigmoid(tf.reshape(
                expert_activations_mru,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        with tf.name_scope('Gate_Luke'):
            gate_activations_luke = slim.fully_connected(
                hidden,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates_luke")
            gating_distribution_luke = tf.nn.softmax(tf.reshape(
                gate_activations_luke,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        with tf.name_scope('Expert_Luke'):
            expert_activations_lvl1_luke = slim.fully_connected(
                hidden,
                vocab_size * num_mixtures,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts_lvl1_luke")

            expert_activations_luke = slim.fully_connected(
                expert_activations_lvl1_luke,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts_luke")

            expert_distribution_luke = tf.nn.sigmoid(tf.reshape(
                expert_activations_luke,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        with tf.name_scope('Gate_Amir'):
            gate_activations_amir = slim.fully_connected(
                hidden,
                vocab_size * (num_mixtures + 1),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates_amir")
            gating_distribution_amir = tf.nn.softmax(tf.reshape(
                gate_activations_amir,
                [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        with tf.name_scope('Expert_Amir'):
            expert_activations_lvl1_amir = slim.fully_connected(
                hidden,
                vocab_size * num_mixtures,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts_lvl1_amir")

            expert_activations_amir = slim.fully_connected(
                expert_activations_lvl1_amir,
                vocab_size * num_mixtures,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="experts_amir")

            expert_distribution_amir = tf.nn.sigmoid(tf.reshape(
                expert_activations_amir,
                [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        with tf.name_scope('Mrudula'):
            final_probabilities_by_class_and_batch_mru = tf.reshape(tf.reduce_sum(
                gating_distribution_mru[:, :num_mixtures] * expert_distribution_mru, 1),
                [-1, 1])
        with tf.name_scope('Luke'):
            final_probabilities_by_class_and_batch_luke = tf.reshape(tf.reduce_sum(
                gating_distribution_luke[:, :num_mixtures] * expert_distribution_luke, 1),
                [-1, 1])
        with tf.name_scope('Amir'):
            final_probabilities_by_class_and_batch_amir = tf.reshape(tf.reduce_sum(
                gating_distribution_amir[:, :num_mixtures] * expert_distribution_amir, 1),
                [-1, 1])

        handshaking = tf.concat([final_probabilities_by_class_and_batch_mru,
                                 final_probabilities_by_class_and_batch_luke,
                                 final_probabilities_by_class_and_batch_amir],
                                axis=1)

        with tf.name_scope('Das_Korinthenkacker'):
            gate_activations_das_korinthenkacker = slim.fully_connected(
                hidden,
                vocab_size * (4),
                activation_fn=None,
                biases_initializer=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope="gates_das_korinthenkacker")

            gate_activations_das_korinthenkacker = tf.reshape(
                gate_activations_das_korinthenkacker,
                [-1, 4])
            das_korinthenkacker = tf.nn.softmax(gate_activations_das_korinthenkacker)

            final_probabilities_by_class_and_batch = tf.reduce_sum(
                das_korinthenkacker[:, :3] * handshaking, 1)
            final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                             [-1, vocab_size])

        return {"predictions": final_probabilities}
