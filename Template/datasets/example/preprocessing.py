from .logger import set_logger
from sklearn import preprocessing
from ..utils import shaping_utils
import numpy as np


def preprocess_dataset(X, Y, n_steps):
  '''
  Return preprocessed dataset from raw dataset.
  Returns:
    - dataset (list): X (np.arr), Y (np.arr)
  '''

  set_logger.info("Starting preprocessing...")

  return preprocess_features(
      *preprocess_labels(*segment_sequences(X, Y, n_steps)))


def preprocess_features(X, Y):
  '''
  Scale the feature vectors using scikit preprocessing.
  '''

  X = np.array(X, dtype=np.float32)
  old_shape = X.shape
  X = X.reshape((-1, X.shape[2]))
  assert(len(X.shape) == 2)  # Double check that X is 2d.

  X = preprocessing.maxabs_scale(X, copy=False)
  X = X.reshape(old_shape)
  return X, Y


def preprocess_labels(X, Y):
  '''
  Numpify labels.
  '''

  return X, np.array(Y, dtype=np.float32)


def segment_sequences(X, Y, n_steps):
  """
  Segment sequence features into segments of uniform length.
  """

  new_X = []
  new_Y = []

  for i in range(len(X)):
    segments = shaping_utils.segment_vector(np.array(X[i]), n_steps)
    new_X += segments
    new_Y += len(segments) * [Y[i]]

  set_logger.debug("Sequence segmentation complete.")
  set_logger.debug("Average segments per sequence: " +
                   str(len(new_X) / len(X)))

  return np.array(new_X), np.array(new_Y)

