from ..utils.network_utils import download_file
from . import config, preprocess_file
import pickle


def main():
  '''
  Main function of this download module.
  Attempts to load existing dataset and if fails, downloads raw dataset and
  saves a preprocessed copy.
  '''

  try:
    pickle.load(open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, "rb"))
    print("Dataset already exists and test load succeded")

  except (OSError, IOError) as e:
    download_file(config.DOWNLOAD_URL, config.DUMPS_DIR + config.RAW_SAVE_NAME)
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, 'wb') as f:
      pickle.dump(preprocess_file(config.DUMPS_DIR + config.RAW_SAVE_NAME), f)

  return None


if __name__ == "__main__":
  main()
