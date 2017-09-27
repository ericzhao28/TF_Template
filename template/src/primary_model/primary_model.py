from . import config
from ..model_base import Base, SequenceLayers
import tensorflow as tf
from .logger import model_logger


class Primary(Base, SequenceLayers):
  def __init__(self, sess, data_config, model_logger=model_logger):
    model_logger.info('Instantiated primary model')
    self.logger = model_logger
    Base.__init__(self, sess, config, data_config, model_logger)

  def build_model(self):
    '''
    Build the primary model (default seq architecture).
    '''

    self.logger.info('Building model...')

    self.x = tf.placeholder(
        tf.float32, (config.BATCH_SIZE, config.N_STEPS, config.N_FEATURES))
    self.target = tf.placeholder(tf.float32,
                                 (config.BATCH_SIZE, config.N_CLASSES))

    encoder_config = {
        'n_batches': config.BATCH_SIZE,
        'n_steps': config.N_STEPS,
        'n_features': config.N_FEATURES,
        'h_gru': config.LAYERS['h_gru'],
        'h_att': config.LAYERS['h_att'],
        'h_dense': config.ENCODED_DIM
    }
    encoded_state, encoded_seq = self._attention_encoder_layer(
        self.x, "encoder", encoder_config)

    predictor_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.ENCODED_DIM,
        'n_classes': config.N_CLASSES
    }
    self.prediction = self._prediction_layer(
        encoded_state,
        'predictor',
        predictor_config)

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

    self.logger.info('Model built.')

    return self

