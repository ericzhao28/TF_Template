from . import config
import pickle


def load():
  '''
  Loads preprocessed data dump if possible.
  '''
  try:
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, "rb") as f:
      return pickle.load(f)

  except (OSError, IOError) as e:
    print("Dataset does not exist")
    return None

