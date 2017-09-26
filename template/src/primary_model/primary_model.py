from . import config
from ..model_base import Base, SequenceLayers
import tensorflow as tf


class Primary(Base, SequenceLayers):
  def __init__(self, sess, data_config):
    Base.__init__(self, sess, config, data_config)

  def build_model(self):
    self.x = tf.placeholder(tf.float32, (config.BATCH_SIZE, config.N_STEPS,
                                         config.N_FEATURES))
    self.target = tf.placeholder(tf.float32, (config.BATCH_SIZE,
                                              config.N_CLASSES))

    encoder_config = {
        'n_batches': config.BATCH_SIZE,
        'n_steps': config.N_STEPS,
        'n_features': config.N_FEATURES,
        'h_gru': config.LAYERS['h_gru'],
        'h_att': config.LAYERS['h_att'],
        'h_dense': config.ENCODED_DIM
    }

    encoded_state, encoded_seq = self._attention_encoder_layer(
        self.embedded_x, "encoder", encoder_config)

    # Get entities predictions
    predictor_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.ENCODED_DIM,
        'n_classes': config.N_CLASSES
    }
    self.prediction = self._prediction_layer(
        encoded_state,
        'predictor',
        predictor_config)

    # Get loss and optimizer
    self.loss, self.acc = self._define_optimization_vars(
        self.target,
        self.prediction,
        None)
    optimizer = tf.train.AdamOptimizer()
    self.optim = optimizer.minimize(
        self.loss,
        var_list=tf.trainable_variables(),
        global_step=self.global_step)
    self.summary_op = self._summaries()

    return self

