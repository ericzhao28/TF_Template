from . import config
from .logger import set_logger
from ..utils import shaping_utils
import numpy as np


def load(test_size=None):
  '''
  Loads dataset.
  '''

  try:
    #######################################
    ### Dataset load.
    with open(config.EXAMPLE_FEATURES_PATH, 'rb') as f:
      set_logger.info("Example features exist. Processing...")
      X = np.load(f)
    with open(config.EXAMPLE_LABELS_PATH, 'rb') as f:
      set_logger.info("Example labels exist. Processing...")
      Y = np.load(f)

    # Shuffle dataset
    X, Y = shaping_utils.shuffle_twins(X, Y)

    if test_size is None:
      return X, Y
    else:
      return (X[test_size:], Y[test_size:]), (X[:test_size], Y[:test_size])
    #######################################

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None

