from ..src import sequential
import numpy as np


def test_generate():
  '''
  Test generate sequential data
  '''

  X, Y = sequential.generate()
  assert(np.array(X).shape == (1808, 5))
  assert(np.array(Y).shape == (1808,))

