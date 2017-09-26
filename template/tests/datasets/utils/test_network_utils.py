from ....datasets.utils import network_utils
from . import config


def test_download_file():
  network_utils.download_file("http://x.com/", config.DUMPS_DIR + "x.html")
  with open(config.DUMPS_DIR + "x.html", 'rb') as myfile:
    data = str(myfile.read().decode())
  assert(data == "x")
