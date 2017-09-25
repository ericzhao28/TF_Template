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
  X = featurize_csv(file_path)
  Y = label_csv(file_path)
  return {"X": X, "Y": Y}


def label_csv(file_path):
  '''
  Load in a CSV and parse for labels.
  Args:
    - file_path (str): path to raw dataset file
  Returns:
    - Y: np.arr((n_points, n_classes), float32)
  '''
  Y = []
  with open(file_path, 'r') as f:
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        continue
      for j, value in enumerate(row):
        if headers_key[j] == config.label_field:
          Y.append(shaping_utils.one_hot(value, config.label_classes))

  return np.array(Y, dtype=np.float32)


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

