import tensorflow as tf
import numpy as np
from .standard_layers import StandardLayers
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(4)
np.random.seed(4)


class Base(StandardLayers):
  '''
  Base model featuring useful Tensorflow utilities.
  '''

  def __init__(self, sess, model_config, data_config, model_logger):
    '''
    Initiate base model.
    Args:
      - sess: tf.Session().
      - model_config: module with model config.
      - data_config: module with data config.
      - model_logger: custom logger handler.
    '''

    self.sess = sess
    self.data_config = data_config
    self.model_config = model_config
    self.saver = None
    self.model_name = "default.model"
    self.global_step = tf.Variable(0,
                                   dtype=tf.int32,
                                   trainable=False,
                                   name='global_step')

  def initialize(self):
    '''
    Initializes model.
    Builds model -> starts summary writer -> global vars init.
    '''

    self.logger.debug('Initializing model...')
    self.build_model()

    self.logger.debug('Model built. Initializing model writer...')
    self.writer = tf.summary.FileWriter(self.data_config.GRAPHS_DIR,
                                        self.sess.graph)

    self.logger.debug('Writer initialized. Initializing TF graph...')
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()
    self.logger.debug('TF graph initialized.')

    self.logger.info('Model initialized.')

  def save(self, global_step=None):
    '''
    Save the current variables in graph.
    Optional option to save for global_step (used in Train).
    '''

    self.logger.debug('Saving model...')
    if self.saver is None:
      self.logger.debug('Saver not initiated, creating new model Saver.')
      self.saver = tf.train.Saver(tf.global_variables())

    if global_step is None:
      self.saver.save(self.sess,
                      self.data_config.CHECKPOINTS_DIR + self.model_name)
      self.logger.debug('Saved with no global step.')
    else:
      self.saver.save(self.sess,
                      self.data_config.CHECKPOINTS_DIR + self.model_name,
                      global_step=self.global_step)
      self.logger.debug('Saved with global step.')

    self.logger.info('Model saved.')

  def restore(self, resume=True):
    '''
    Restore TF computation graph from saved checkpoint.
    Args:
      - resume (bool): resume last checkpoint or restore standard
                       save file.
    '''

    self.logger.debug('Restoring model...')

    if self.saver is None:
      self.logger.debug('Saver not initiated, creating new model Saver.')
      self.saver = tf.train.Saver(tf.global_variables())

    if resume:
      self.logger.debug('Resume enabled. Finding newest model checkpoint.')

      ckpt = tf.train.latest_checkpoint(self.data_config.CHECKPOINTS_DIR)
      if ckpt:
        self.logger.debug('Model checkpoint found. Restoring...')
        self.saver.restore(self.sess, ckpt)
        self.logger.info('Model restored. Resuming from checkpoint.')
        return True
      else:
        self.logger.error('Resume enabled but no model checkpoints found. \
                           \n Terminating...')
        raise ValueError()
    else:
      self.logger.debug('Resume disabled. Restoring from default save...')
      self.saver.restore(
          self.sess, self.data_config.CHECKPOINTS_DIR + self.model_name)
      self.logger.info('Model restored.')

  def train(self, X, Y):
    '''
    Run model training. Model must have been initialized.
    Args:
      X (np.arr): featured data. Assuming len(X) > batch size.
      Y (np.arr): labels. Assuming len(Y) > batch size.
    '''

    self.logger.info('Starting model training...')

    for j in range(self.model_config.ITERATIONS):
      for i in range(0, len(X) + 1 - self.model_config.BATCH_SIZE,
                     self.model_config.BATCH_SIZE):
        feed_dict = {
            self.x: X[i:i + self.model_config.BATCH_SIZE],
            self.target: Y[i:i + self.model_config.BATCH_SIZE]
        }
        _, acc, loss, summary = self.sess.run(
            [self.optim, self.acc, self.loss, self.summary_op],
            feed_dict=feed_dict)
      if j % 10 == 0:
        self.logger.info("Epoch: " + str(j) + " has loss: " +
                         str(float(loss)) + " and accuracy: " +
                         str(float(acc)))
        self.save(self.global_step)
      self.writer.add_summary(summary, global_step=j)

    self.logger.info('Model finished training!')

  def predict(self, X):
    '''
    Predict classifications for new inputs.
    Args:
      X (np.arr): featured data. Assuming len(X) > batch size.
                  Note that left over data from batching are
                  NOT calculated. So pad batches beforehand.
    Returns:
      predictions (list of np.arr): flat list of predictions.
    '''

    self.logger.info('Starting model predictions...')
    predictions = []
    for i in range(0, len(X) + 1 - self.model_config.BATCH_SIZE,
                   self.model_config.BATCH_SIZE):
      feed_dict = {
          self.x: X[i:i + self.model_config.BATCH_SIZE]
      }
      predictions += list(self.sess.run([self.prediction],
                          feed_dict=feed_dict)[0])
    self.logger.info('Model finished predicting!')
    return predictions

  def shuffle_and_partition(self, X, Y, n_test, n_val):
    '''
    Shuffle and partition input data.
    Args:
      - X (np.array): X data
      - Y (np.array): target to be shuffled in sync
      - n_test (int): number of test points
      - n_val (int): number of validation points
    Return:
      - result:
        {"train": {"X": np.arr, "Y": np.arr},
        "test": {"X": np.arr, "Y": np.arr},
        "val": {"X": np.arr, "Y": np.arr}}.
    '''

    n_train = X.shape[0] - n_test - n_val
    self.logger.debug('Shuffling and partitioning data...')

    self.logger.debug('Shuffling X, Y in sync...')
    p = np.random.permutation(X.shape[0])
    shuffled_X, shuffled_Y = X[p], Y[p]
    del(p)
    del(X)
    del(Y)
    self.logger.debug('Finished shuffling dataset.')

    # Structured as: [train, test, val]
    self.logger.debug('Partitioning with training size: ' + str(n_train) +
                      ' test size: ' + str(n_test) + ' and val size: ' +
                      str(n_val) + '...')
    self.logger.debug('First partitioning X...')
    train_X = shuffled_X[:n_train]
    test_X = shuffled_X[n_train:(n_train + n_test)]
    val_X = shuffled_X[(n_train + n_test):]
    del(shuffled_X)

    self.logger.debug('Finished partitioning X. Now partitioning Y...')
    train_Y = shuffled_Y[:n_train]
    test_Y = shuffled_Y[n_train:(n_train + n_test)]
    val_Y = shuffled_Y[(n_train + n_test):]
    del(shuffled_Y)
    self.logger.debug('Finished partitioning Y.')

    self.logger.info('Finished shuffling and partitioning.')

    return {"train": {"X": train_X, "Y": train_Y},
            "test": {"X": test_X, "Y": test_Y},
            "val": {"X": val_X, "Y": val_Y}}

