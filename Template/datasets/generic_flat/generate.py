from . import config, preprocess_file
import pickle
from .logger import set_logger


def main():
  '''
  Main function of this download module.
  Attempts to load existing dataset and if fails, downloads raw dataset and
  saves a preprocessed copy.
  '''

  set_logger.info("Beginning dataset generation")
  with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, 'wb') as f:
    pickle.dump(preprocess_file(config.DUMPS_DIR + config.RAW_SAVE_NAME), f)
    set_logger.info("Dataset pickle loaded and dumped.")


if __name__ == "__main__":
  main()
