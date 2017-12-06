import tensorflow as tf
import numpy as np
from .standard_layers import StandardLayers


class Base(StandardLayers):
  '''
  Base model featuring useful Tensorflow utilities.
  '''

  def __init__(self, sess, flags, logger, global_step=0,
               model_name="default.model"):
    '''
    Initiate base model.
    Args:
      - sess: tf.Session().
      - flags: flags with model config.
      - logger: custom logger handler.
      - global_step: global_step of the model.
      - model_name: name of the model.
    '''

    self.sess = sess
    self.flags = flags
    self.logger = logger
    self.model_name = model_name
    self.min_acc = None
    self.global_step = tf.Variable(global_step,
                                   dtype=tf.int32,
                                   trainable=False,
                                   name='global_step')

  def initialize(self):
    '''
    Builds model -> starts summary writer -> global vars init.
    '''

    self.logger.debug('Initializing model...')
    self.build_model()
    self.logger.debug('Model built. Initializing model writer...')
    self.train_writer = tf.summary.FileWriter(self.flags.graphs_train_dir,
                                              self.sess.graph)
    self.test_writer = tf.summary.FileWriter(self.flags.graphs_test_dir,
                                             self.sess.graph)
    self.logger.debug('Writer initialized. Initializing TF graph...')
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()
    self.logger.debug('TF graph initialized.')

    self.logger.info(self.model_name + ' initialized.')

  @property
  def saver(self):
    try:
      return self.tf_saver
    except AttributeError:
      self.logger.debug('Saver not initiated, creating new model Saver.')
      self.tf_saver = tf.train.Saver(tf.global_variables())
      return self.tf_saver

  def save(self, global_step):
    '''
    Save the current variables in graph.
    '''

    self.logger.debug('Saving model...')
    self.saver.save(self.sess,
                    self.flags.checkpoints_dir + self.model_name,
                    global_step=self.global_step)
    self.logger.debug('Saved with global step.')

    self.logger.info(self.model_name + ' saved.')

  def restore(self, ckpt):
    '''
    Restore TF computation graph from saved checkpoint.
    '''

    self.logger.debug('Restoring model...')
    self.saver.restore(self.sess, ckpt)
    self.logger.info(self.model_name + ' restored.')

  def train(self, X, Y, report_func):
    '''
    Run model training. Model must have been initialized.
    Args:
      X (np.arr): featured data. Assuming len(X) > batch size.
      Y (np.arr): labels. Assuming len(Y) > batch size.
      report_func: function to call whenever reporting is triggered.
                   takes two arguments: sub_epoch (self, int).
    '''

    self.logger.info(self.model_name + ' starting training...')

    for epoch in range(self.flags.n_epochs):
      for x_batch, y_batch in self.yield_batch(X, Y):
        feed_dict = {
            self.x: x_batch,
            self.target: y_batch
        }
        _, loss = self.sess.run(
            [self.optim, self.loss],
            feed_dict=feed_dict)

        iteration = tf.train.global_step(self.sess, self.global_step)
        if iteration % self.flags.s_report_interval == 0:
          report_func(self, iteration)

    self.logger.info(self.model_name + ' finished training!')

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

    self.logger.debug('Starting model predictions...')
    predictions = []
    for x_batch in self.yield_batch(X):
      feed_dict = {
          self.x: x_batch,
      }
      predictions += list(self.sess.run([self.prediction],
                          feed_dict=feed_dict)[0])
    self.logger.debug('Model finished predicting!')
    return np.array(predictions)

  def evaluate(self, X, Y, prefix=""):
    '''
    Run model evaluation. Model must have been initialized.
    Args:
      X (np.arr): featured data. Assuming len(X) > batch size.
      Y (np.arr): labels. Assuming len(Y) > batch size.
      Prefix (str): prefix for summary tags.
    '''

    all_loss = []
    all_tpr = []
    all_fpr = []
    all_acc = []

    for x_batch, y_batch in self.yield_batch(X, Y):
      feed_dict = {
          self.x: x_batch,
          self.target: y_batch
      }

      tpr, fpr, acc, loss = self.sess.run(
          [self.tpr, self.fpr, self.acc, self.loss],
          feed_dict=feed_dict)
      tpr = float(tpr)
      fpr = float(fpr)
      all_loss.append(loss)
      all_tpr.append(tpr)
      all_fpr.append(fpr)
      all_acc.append(acc)

    # Average metrics
    avg_loss = np.mean(all_loss)
    avg_tpr = np.mean(all_tpr)
    avg_fpr = np.mean(all_fpr)
    avg_acc = np.mean(all_acc)

    # Create summaries
    summary = tf.Summary()
    summary.value.add(tag="%s/Accuracy" % prefix,
                      simple_value=avg_acc)
    summary.value.add(tag="%s/Loss" % prefix, simple_value=avg_loss)
    if all_fpr:
      summary.value.add(tag="%s/FPR" % prefix, simple_value=avg_fpr)
    if all_tpr:
      summary.value.add(tag="%s/TPR" % prefix, simple_value=avg_tpr)

    return avg_loss, avg_acc, avg_tpr, avg_fpr, summary

  def yield_batch(self, X, Y=None):
    """
    Break arrays into batches.
    Args:
      X (np.arr): mandatory first arr
      Y (np.arr): optional second arr
    """

    for i in range(0, X.shape[0] + 1 - self.flags.s_batch,
                   self.flags.s_batch):
      if Y is not None:
        yield X[i:(i + self.flags.s_batch)], \
            Y[i:(i + self.flags.s_batch)]
      else:
        yield X[i:(i + self.flags.s_batch)]

