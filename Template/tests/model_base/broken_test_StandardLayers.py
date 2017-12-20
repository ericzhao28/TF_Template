from ...src.model_base import StandardLayers
import tensorflow as tf


def test_prediction():
  X = tf.placeholder(tf.float32, (3, 7))
  predictor_config = {
      'n_batches': 3,
      'n_input': 7,
      'n_classes': 5
  }
  prediction = StandardLayers()._prediction_layer(
      X, 'predictor', predictor_config)

  assert(prediction.shape == (3, 5))
  assert(prediction.dtype == tf.float32)


def test_optimizations():
  prediction = tf.placeholder(tf.float32, (3, 5))
  target = tf.placeholder(tf.float32, (3, 5))
  loss, acc = StandardLayers()._define_optimization_vars(
      target, prediction, [1, 1, 0, 0, 1], 0.1)

  assert(loss.shape == [])
  assert(loss.dtype == tf.float32)
  assert(acc.shape == [])
  assert(acc.dtype == tf.float32)


def test_binary_metrics():
  prediction = tf.placeholder(tf.float32, (3, 5))
  target = tf.placeholder(tf.float32, (3, 5))
  TPR, FPR, acc = StandardLayers()._define_binary_metrics(
      target, prediction)

  assert(TPR.shape == [])
  assert(TPR.dtype == tf.float32)
  assert(FPR.shape == [])
  assert(FPR.dtype == tf.float32)

