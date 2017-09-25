from ..src import sequential
import numpy as np


def test_generate():
  '''
  Test generate sequential data
  '''

  X, Y = sequential.generate()
  assert(np.array(X).shape == (200 * 20, 5))
  assert(np.array(Y).shape == (200 * 20,))

