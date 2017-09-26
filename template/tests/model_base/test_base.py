from ...src.primary_model import Primary
from ...datasets.test_set import config as data_config
from ...datasets.test_set import load
import numpy as np
import tensorflow as tf


def test_build():
  tf.reset_default_graph()
  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    model.initialize()
  tf.reset_default_graph()


def test_train():
  X, Y = load()
  tf.reset_default_graph()
  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    model.initialize()
    model.train(X[10:], Y[10:])

    predictions = model.predict(X[:10])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(Y[:10], 1))
    acc = np.reduce_mean(np.cast(correct, tf.float32))

    assert(acc > 0.8)
  tf.reset_default_graph()


def test_load():
  X, Y = load()
  tf.reset_default_graph()
  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    model.initialize()
    model.train(X[10:], Y[10:])
    model.save()

    predictions = model.predict(X[:10])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(Y[:10], 1))
    acc = np.reduce_mean(np.cast(correct, tf.float32))
    assert(acc > 0.8)

  tf.reset_default_graph()
  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    model.initialize()
    model.restore()

    predictions = model.predict(X[:10])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(Y[:10], 1))
    new_acc = np.reduce_mean(np.cast(correct, tf.float32))

    assert(acc == new_acc)
  tf.reset_default_graph()

