from ..model_base import Base, SequenceLayers
import tensorflow as tf


class FlatModel(Base, SequenceLayers):
  '''
  Model for predicting on flat data.
  Model-specific config requirements:
    BATCH_SIZE
    N_FEATURES
    N_CLASSES
  '''

  def __init__(self, sess, flags, logger, **kargs):
    logger.debug('Instantiated flat model')
    Base.__init__(self, sess, flags, logger, **kargs)

  def build_model(self):
    '''
    Build the flat model.
    '''

    self.logger.debug('Building model...')
    flags = self.flags

    self.x = tf.placeholder(
        tf.float32, (flags.s_batch, flags.n_features))
    self.target = tf.placeholder(tf.float32,
                                 (flags.s_batch, flags.n_classes))

    dense_config = {
        'n_batches': flags.s_batch,
        'n_input': flags.n_features,
        'n_hidden': flags.h_dense,
        'n_output': flags.o_dense
    }
    dense_state = self._dense_layer(
        encoded_state, "dense", dense_config)

    predictor_config = {
        'n_batches': flags.s_batch,
        'n_input': flags.o_dense,
        'n_classes': flags.n_classes
    }
    self.prediction = self._prediction_layer(
        dense_state,
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

