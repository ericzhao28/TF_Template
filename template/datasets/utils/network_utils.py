"""
This module offers miscellanious networking functions.
"""

import requests


def download_file(url, file_path):
  '''
  Generic function to stream-download a large file
  '''
  r = requests.get(url, stream=True)
  with open(file_path, 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024):
      if chunk:
        f.write(chunk)
  return file_path
