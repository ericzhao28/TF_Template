from ..model_base import Base, SequenceLayers
import tensorflow as tf


class FlowAttModel(Base, SequenceLayers):
  '''
  Model for predicting on flows.
  '''

  def __init__(self, sess, flags, logger, **kargs):
    logger.debug('Instantiated flow att model')
    Base.__init__(self, sess, flags, logger, **kargs)

  def build_model(self):
    '''
    Build the flow model.
    '''

    self.logger.debug('Building model...')
    flags = self.flags

    self.x = tf.placeholder(
        tf.float32, (flags.s_batch, flags.n_steps, flags.n_features))
    self.target = tf.placeholder(tf.float32,
                                 (flags.s_batch, flags.n_classes))

    encoder_config = {
        'n_batches': flags.s_batch,
        'n_steps': flags.n_steps,
        'n_features': flags.n_features,
        'h_gru': flags.h_gru,
        'h_att': flags.h_att,
        'h_dense': flags.o_gru
    }
    encoded_state = self._attention_encoder_layer(
        self.x, "encoder", encoder_config)

    dense_config = {
        'n_batches': flags.s_batch,
        'n_input': flags.o_gru,
        'n_hidden': flags.h_dense,
        'n_output': flags.o_dense
    }
    dense_state = self._dense_layer(
        encoded_state, "dense", dense_config)

    double_dense_config = {
        'n_batches': flags.s_batch,
        'n_input': flags.o_dense,
        'n_hidden': flags.h_dense2,
        'n_output': flags.o_dense2
    }
    double_dense_state = self._dense_layer(
        dense_state, "dense2", double_dense_config)

    predictor_config = {
        'n_batches': flags.s_batch,
        'n_input': flags.o_dense2,
        'n_classes': flags.n_classes
    }
    self.prediction = self._prediction_layer(
        double_dense_state,
        'predictor',
        predictor_config)

    self.loss = self._define_optimization_vars(
        self.target,
        self.prediction,
        [1, 1],
        flags.v_regularization
    )
    self.tpr, self.fpr, self.acc = self._define_binary_metrics(
        self.target,
        self.prediction,
    )

    optimizer = tf.train.AdamOptimizer()
    self.optim = optimizer.minimize(
        self.loss,
        var_list=tf.trainable_variables(),
        global_step=self.global_step)

    self.logger.debug('Model built.')

    return self

