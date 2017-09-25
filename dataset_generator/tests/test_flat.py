from ..src import flat
import numpy as np


def test_generate():
  '''
  Test generate flat data
  '''

  X, Y = flat.generate()
  assert(np.array(X).shape == (200, 4))
  assert(np.array(Y).shape == (200,))

