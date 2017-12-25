import random
from ..utils.network_utils import upload_file
from . import config, preprocess_dataset
from .logger import set_logger
import numpy as np


random.seed(0)


def main():
  '''
  Generates generic sequential dataset.
  '''

  set_logger.info("Beginning example dataset generation")

  ##############################
  ### Generate full dataset
  X = []
  Y = []
  for i in range(0, 100, 10):
    for j in range(20):
      x = []
      for k in range(random.randrange(7, 12)):
        # Variable length datapoints, 7-12 long.
        x.append([
            random.randrange(i + k, i + k + 2, 0.01),
            random.randrange(0, 100, 0.01),
            random.randrange(1, 100, 0.01),
            random.randrange(i + k, i + k + 3, 0.01)
        ])
      # Total of 200 data points generated.
      X.append(x)
      Y.append(i)
  ##############################

  ##############################
  ### Preprocess dataset
  X, Y = preprocess_dataset(X, Y, config.n_steps)
  set_logger.info("Dataset preprocessed.")
  with open(config.EXAMPLE_FEATURES_PATH, 'wb') as f:
    np.save(f, np.array(X, dtype=np.float32))
    set_logger.info("Example features dumped.")
  del(X)
  with open(config.EXAMPLE_LABELS_PATH, 'wb') as f:
    np.save(f, np.array(Y, dtype=np.float32))
    set_logger.info("Example labels dumped.")
  del(Y)
  ##############################

  ##############################
  ### Upload files
  upload_file("datasets", "example_X",
              config.EXAMPLE_FEATURES_PATH)
  upload_file("datasets", "example_Y",
              config.EXAMPLE_LABELS_PATH)
  ##############################

  return None


if __name__ == "__main__":
  main()

