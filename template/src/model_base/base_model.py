import tensorflow as tf
from .standard_layers import StandardLayers

tf.logging.set_verbosity(tf.logging.ERROR)


class Base(StandardLayers):
  '''
  Base model that handles save, restore, load, train functionalities
  '''

  def __init__(self, sess, model_config, data_config):
    tf.set_random_seed(4)
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
    Initialize models: builds model, loads data, initializes variables
    '''
    self.build_model()
    self.writer = tf.summary.FileWriter(self.data_config.GRAPHS_DIR,
                                        self.sess.graph)
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()

  def save(self, global_step=None):
    '''
    Save the current variables in graph.
    Optional option to save for global_step (used in Train)
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())

    if global_step is None:
      self.saver.save(self.sess,
                      self.data_config.CHECKPOINTS_DIR + self.model_name)
    else:
      self.saver.save(self.sess,
                      self.data_config.CHECKPOINTS_DIR + self.model_name,
                      global_step=self.global_step)

  def restore(self, resume=False):
    '''
    Load saved variable values into graph
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())

    if resume:
      ckpt = tf.train.get_checkpoint_state(self.data_config.CHECKPOINTS_DIR +
                                           'checkpoint')
      if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      return

    self.saver.restore(self.sess,
                       self.data_config.CHECKPOINTS_DIR + self.model_name)

  def train(self, X, Y):
    '''
    Run model training. Model must have been initialized.
    '''
    for _ in range(self.model_config.ITERATIONS):
      for i in range(0, len(X), self.model_config.BATCH_SIZE):
        feed_dict = {
            self.x: X[i:i + self.model_config.BATCH_SIZE]
        }
        _, acc, loss, summary = self.sess.run(
            [self.optim, self.acc, self.loss, self.summary_op],
            feed_dict=feed_dict)
        i += 1
        print("Epoch:", i, "has loss:", loss, "and accuracy:", acc)
        self.writer.add_summary(summary, global_step=self.global_step)

  def predict(self, X):
    '''
    Predict classifications for new inputs
    '''
    predictions = []
    for i in range(0, len(X), self.model_config.BATCH_SIZE):
      feed_dict = {
          self.x: X[i:i + self.model_config.BATCH_SIZE]
      }
      predictions += list(self.sess.run([self.prediction],
                          feed_dict=feed_dict))
    return predictions

