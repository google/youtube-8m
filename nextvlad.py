from frame_level_models import *

flags.DEFINE_integer("nextvlad_cluster_size", 64, "Number of units in the NeXtVLAD cluster layer.")
flags.DEFINE_integer("nextvlad_hidden_size", 1024, "Number of units in the NeXtVLAD hidden layer.")

flags.DEFINE_integer("groups", 8, "number of groups in VLAD encoding")
flags.DEFINE_float("drop_rate", 0.5, "dropout ratio after VLAD encoding")
flags.DEFINE_integer("expansion", 2, "expansion ratio in Group NetVlad")
flags.DEFINE_integer("gating_reduction", 8, "reduction factor in se context gating")

flags.DEFINE_integer("mix_number", 3, "the number of gvlad models")
flags.DEFINE_float("cl_temperature", 2, "temperature in collaborative learning")
flags.DEFINE_float("cl_lambda", 1.0, "penalty factor of cl loss")


class NeXtVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, is_training=True, expansion=2, groups=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

    def forward(self, input, mask=None):
        input = slim.fully_connected(input, self.expansion * self.feature_size, activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(input, self.groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, self.max_frames*self.groups, 1])
        tf.summary.histogram("sigmoid_attention", attention)
        feature_size = self.expansion * self.feature_size // self.groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion*self.feature_size, self.groups*self.cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        # tf.summary.histogram("cluster_weights", cluster_weights)
        reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="cluster_bn",
            fused=False)

        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)
        activation = tf.multiply(activation, attention)
        # tf.summary.histogram("cluster_output", activation)
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
        vlad = slim.batch_norm(vlad,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="vlad_bn",
                fused=False)

        return vlad


class NeXtVLADModel(models.BaseModel):
    """Creates a NeXtVLAD based model.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     groups=None,
                     expansion=None,
                     drop_rate=None,
                     gating_reduction=None,
                     **unused_params):
        cluster_size = cluster_size or FLAGS.nextvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.nextvlad_hidden_size
        gating_reduction = gating_reduction or FLAGS.gating_reduction
        groups = groups or FLAGS.groups
        drop_rate = drop_rate or FLAGS.drop_rate
        expansion = expansion or FLAGS.expansion

        mask = tf.sequence_mask(num_frames, 300, dtype=tf.float32)
        max_frames = model_input.get_shape().as_list()[1]
        video_nextvlad = NeXtVLAD(1024, max_frames, cluster_size, is_training, groups=groups, expansion=expansion)
        audio_nextvlad = NeXtVLAD(128, max_frames, cluster_size // 2, is_training, groups=groups // 2, expansion=expansion)

        with tf.variable_scope("video_VLAD"):
            vlad_video = video_nextvlad.forward(model_input[:, :, 0:1024], mask=mask)

        with tf.variable_scope("audio_VLAD"):
            vlad_audio = audio_nextvlad.forward(model_input[:, :, 1024:], mask=mask)

        vlad = tf.concat([vlad_video, vlad_audio], 1)

        if drop_rate > 0.:
            vlad = slim.dropout(vlad, keep_prob=1. - drop_rate, is_training=is_training, scope="vlad_dropout")

        vlad_dim = vlad.get_shape().as_list()[1]
        print("VLAD dimension", vlad_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        # activation = tf.nn.relu(activation)

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [hidden1_size, hidden1_size // gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [hidden1_size // gating_reduction, hidden1_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)
        tf.summary.histogram("final_gates", gates)

        activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class MixNeXtVladModel(models.BaseModel):

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     mix_number=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     groups=None,
                     expansion=None,
                     drop_rate=None,
                     gating_reduction=None,
                     **unused_params):
        cluster_size = cluster_size or FLAGS.nextvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.nextvlad_hidden_size
        gating_reduction = gating_reduction or FLAGS.gating_reduction
        groups = groups or FLAGS.groups
        drop_rate = drop_rate or FLAGS.drop_rate
        mix_number = mix_number or FLAGS.mix_number
        expansion = expansion or FLAGS.expansion
        mask = tf.sequence_mask(num_frames, 300, dtype=tf.float32)

        max_frames = model_input.get_shape().as_list()[1]

        ftr_mean = tf.reduce_mean(model_input, axis=1)
        ftr_mean = slim.batch_norm(ftr_mean,
                                   center=True,
                                   scale=True,
                                   fused=True,
                                   is_training=is_training,
                                   scope="mix_weights_bn")
        mix_weights = slim.fully_connected(ftr_mean, mix_number, activation_fn=None,
                                           weights_initializer=slim.variance_scaling_initializer(),
                                           scope="mix_weights")
        mix_weights = tf.nn.softmax(mix_weights, axis=-1)
        tf.summary.histogram("mix_weights", mix_weights)

        results = []
        for n in range(mix_number):
            with tf.variable_scope("branch_%d"%n):
                res = self.nextvlad_model(video_ftr=model_input[:, :, 0:1024], audio_ftr=model_input[:, :, 1024:], vocab_size=vocab_size,
                                          max_frames=max_frames, cluster_size=cluster_size, groups=groups, expansion=expansion,
                                          drop_rate=drop_rate, hidden1_size=hidden1_size, is_training=is_training,
                                          gating_reduction=gating_reduction, mask=mask, **unused_params)
                results.append(res)

        aux_preds = [res["predictions"] for res in results]
        logits = [res["logits"] for res in results]
        logits = tf.stack(logits, axis=1)

        mix_logit = tf.reduce_sum(tf.multiply(tf.expand_dims(mix_weights, -1), logits), axis=1)

        pred = tf.nn.sigmoid(mix_logit)

        if is_training:
            rank_pred = tf.expand_dims(tf.nn.softmax(tf.div(mix_logit, FLAGS.cl_temperature), axis=-1), axis=1)
            aux_rank_preds = tf.nn.softmax(tf.div(logits, FLAGS.cl_temperature), axis=-1)
            epsilon = 1e-8
            kl_loss = tf.reduce_sum(rank_pred * (tf.log(rank_pred + epsilon) - tf.log(aux_rank_preds + epsilon)),
                                    axis=-1)

            regularization_loss = FLAGS.cl_lambda * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1), axis=-1)

            return  {"predictions": pred,
                     "regularization_loss": regularization_loss,
                     "aux_predictions": aux_preds}
        else:
            return {"predictions": pred}
            # return {"predictions": results[0]["predictions"]}

    def nextvlad_model(self, video_ftr, audio_ftr, vocab_size, max_frames,
                       cluster_size, groups, drop_rate, hidden1_size,
                       is_training, gating_reduction, mask, expansion,
                       **unused_params):
        video_vlad = NeXtVLAD(1024, max_frames, cluster_size=cluster_size, groups=groups, expansion=expansion,
                              is_training=is_training)
        audio_vlad = NeXtVLAD(128, max_frames, cluster_size=cluster_size // 2, groups=groups // 2, expansion=expansion,
                              is_training=is_training)

        with tf.variable_scope("video_vlad"):
            video_ftr = video_vlad.forward(video_ftr, mask=mask)
        with tf.variable_scope("audio_vlad"):
            audio_ftr = audio_vlad.forward(audio_ftr, mask=mask)

        vlad = tf.concat([video_ftr, audio_ftr], 1)

        if drop_rate > 0.:
            vlad = slim.dropout(vlad, keep_prob=1. - drop_rate, is_training=is_training, scope="vlad_dropout")

        vlad_dim = vlad.get_shape().as_list()[1]
        print("VLAD dimension", vlad_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden1_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [hidden1_size, hidden1_size // gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [hidden1_size // gating_reduction, hidden1_size],
                                           initializer=slim.variance_scaling_initializer())
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)
