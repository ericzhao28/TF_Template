from . import config
from ..utils import csv_utils, shaping_utils
import numpy as np
import csv


def preprocess_file(file_path):
  '''
  Return preprocessed dataset from raw data file.
  Args:
    - file_path (str): path to raw dataset file
  Returns:
    - dataset (dict): {"X": np.arr, "Y": np.arr}
  '''
  X_flat = featurize_csv(file_path)
  metadata = metadatize_csv(file_path)
  X, Y = segment(X_flat, metadata)

  return {"X": X, "Y": Y}


def segment(X_flat, metadata):
  raw = {}
  for i in range(len(metadata)):
    if metadata[i]['seq_id'] not in raw:
      raw[metadata[i]['seq_id']] = {"metadatum":metadata[i], "x":[]}
    raw[metadata[i]['seq_id']]['x'].append(X_flat[i])

  X = []
  Y = []
  for seq_id, data in raw:
    X.append(shaping_utils.fix_vector_length(data['x'], config.SEQ_LEN))
    Y.append(data['metadatum']['y'])

  return X, Y


def metadatize_csv(file_path):
  '''
  Load in a CSV and parse for metadatas.
  Args:
    - file_path (str): path to raw dataset file
  Returns:
    - metadata: [{info:a, label:x}... (n_classes)]
  '''
  metadata = []
  with open(file_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        continue
      metadatum = {}
      for j, value in enumerate(row):
        if headers_key[j] == config.label_field:
          metadatum['y'] = shaping_utils.one_hot(value, config.label_classes)
        elif headers_key[j] == config.seq_field:
          metadatum['seq_id'] = value
      metadata.append(metadatum)

  return metadata


def featurize_csv(file_path):
  '''
  Load in a CSV and parse for features.
  Args:
    - file_path (str): path to raw dataset file
  Returns:
    - X: np.arr((n_points, n_features), float32)
  '''
  X = []
  with open(file_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        continue
      X.append(featurize_row(row, headers_key))

  return np.array(X, dtype=np.float32)


def featurize_row(row, headers_key):
  '''
  Featurize a row into a real-valued vector
  Args:
    - row (list of str): row of a csv from csv.reader
    - headers_key (dict): maps pos index in row to field name
  Returns:
    - feature_vector: np.arr((n_features), float32)
  '''
  feature_vector = []
  for i, value in enumerate(row):
    if headers_key[i] in config.numerical_fields:
      feature_vector.append(float(value))
  assert(len(feature_vector) == len(config.numerical_fields))

  return np.array(feature_vector, dtype=np.float32)

