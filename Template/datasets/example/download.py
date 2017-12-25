from ..utils.network_utils import download_file
from . import config


if __name__ == "__main__":
  download_file("datasets", "example_X", config.EXAMPLE_FEATURES_PATH)
  download_file("datasets", "example_Y", config.EXAMPLE_LABELS_PATH)

