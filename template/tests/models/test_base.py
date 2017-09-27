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
  tf.reset_default_graph()
  X, Y = load()

  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    shuffled_dataset = model.shuffle_and_partition(X, Y, 20, 0)
    del(X)
    del(Y)
    model.initialize()
    model.train(shuffled_dataset['train']['X'], shuffled_dataset['train']['Y'])

    predictions = model.predict(shuffled_dataset['train']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['train']['Y'][:len(predictions)], 1))
    acc = np.mean(correct.astype(np.float32))
    print("Train acc:", acc)
    assert(acc >= 0.6)

    predictions = model.predict(shuffled_dataset['test']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['test']['Y'][:len(predictions)], 1))
    acc = np.mean(correct.astype(np.float32))
    print("Test acc:", acc)
    assert(acc >= 0.4)

  tf.reset_default_graph()


def test_checkpointing():
  tf.reset_default_graph()
  X, Y = load()

  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    shuffled_dataset = model.shuffle_and_partition(X, Y, 20, 0)
    del(X)
    del(Y)
    model.initialize()
    model.train(shuffled_dataset['train']['X'], shuffled_dataset['train']['Y'])

    predictions = model.predict(shuffled_dataset['test']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['test']['Y'][:len(predictions)], 1))
    acc = np.mean(correct.astype(np.float32))
    print("Test initial load acc:", acc)
    assert(acc > 0.4)

  tf.reset_default_graph()

  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    model.initialize()
    model.restore()

    predictions = model.predict(shuffled_dataset['test']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['test']['Y'][:len(predictions)], 1))
    new_acc = np.mean(correct.astype(np.float32))
    print("Test loaded acc:", new_acc)
    assert(new_acc > 0.4)

  tf.reset_default_graph()


def test_load():
  tf.reset_default_graph()
  X, Y = load()

  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    shuffled_dataset = model.shuffle_and_partition(X, Y, 20, 0)
    del(X)
    del(Y)
    model.initialize()
    model.train(shuffled_dataset['train']['X'], shuffled_dataset['train']['Y'])
    model.save()

    predictions = model.predict(shuffled_dataset['test']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['test']['Y'][:len(predictions)], 1))
    acc = np.mean(correct.astype(np.float32))
    print("Test initial load acc:", acc)
    assert(acc > 0.4)

  tf.reset_default_graph()

  with tf.Session() as tf_sess:
    model = Primary(tf_sess, data_config)
    model.initialize()
    model.restore()

    predictions = model.predict(shuffled_dataset['test']['X'])
    correct = np.equal(np.argmax(predictions, 1), np.argmax(
        shuffled_dataset['test']['Y'][:len(predictions)], 1))
    new_acc = np.mean(correct.astype(np.float32))
    print("Test loaded acc:", new_acc)
    assert(new_acc > 0.4)

  tf.reset_default_graph()

