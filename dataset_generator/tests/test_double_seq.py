from ..src import double_seq
import numpy as np


def test_generate():
  '''
  Test generate double_seq data
  '''

  X, Y = double_seq.generate()
  assert(np.array(X).shape == (17640, 6))
  assert(np.array(Y).shape == (17640,))

