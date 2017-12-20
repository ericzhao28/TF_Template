from ....datasets.utils import shaping_utils
import numpy as np


def test_build_one_hot():
  """
  Test shaping_utils.build_one_hot(entry, options).
  """

  def __normal_case():
    """
    Handle normal one hot encoding case.
    """

    assert(np.all(shaping_utils.build_one_hot(
        "hello", ["hi", "hello", "hugs"]) == np.array([0, 1, 0])))
    assert(np.all(shaping_utils.build_one_hot(
        "hi", ["hi", "hello", "hugs"]) == np.array([1, 0, 0])))
    assert(np.all(shaping_utils.build_one_hot(
        "hugs", ["hi", "hello", "hugs"]) == np.array([0, 0, 1])))

  def __invalid_case():
    """
    Case that should result in a crash due to invalid
    one-hot options.
    """

    failed = False
    try:
      shaping_utils.build_one_hot("lonely", ["hi", "hello", "hugs"])
    except ValueError:
      failed = True
    assert(failed)

  __normal_case()
  __invalid_case()


def test_fix_vector_length():
  assert(np.all(np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]) ==
         shaping_utils.fix_vector_length(np.full((2, 3), fill_value=1), 4)))
  assert(np.all(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]) ==
         shaping_utils.fix_vector_length(np.full((7, 3), fill_value=1), 4)))
  assert(np.all(np.array([
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1], [1, 1]],
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0], [0, 0]]]) ==
      shaping_utils.fix_vector_length(np.full((3, 4, 2), fill_value=1), 4)))
  assert(np.all(np.array([
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1], [1, 1]],
      [[1, 1], [1, 1], [1, 1], [1, 1]]]) ==
      shaping_utils.fix_vector_length(np.full((5, 4, 2), fill_value=1), 3)))
  assert(np.all(np.array([
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1], [1, 1]],
      [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]]) ==
      shaping_utils.fix_vector_length(np.full((2, 4, 2), fill_value=1), 6)))


def test_segment_vector():
  """
  Test shaping_utils.segment_vector(vector, length).
  """

  vector1 = np.repeat(np.expand_dims(np.arange(30), axis=1), axis=1, repeats=2)

  # Validate we built vector 1 correctly for this test.
  assert(vector1.shape == (30, 2))
  assert(np.all(vector1[1] == [1, 1]))
  assert(np.all(vector1[7] == [7, 7]))

  def __too_short():
    """
    Test segmenting when length > vector.size[0].
    Should result in [x_1, x_2... x_n, 0, 0]
    """

    segments = np.array(shaping_utils.segment_vector(vector1, 32))
    assert(segments.shape == (1, 32, 2))
    assert(np.all(segments == np.pad(
        np.array(vector1, copy=True), [[0, 2], [0, 0]], 'constant',
        constant_values=0)))

  def __exact_cut():
    """
    Test segmenting for exact matches.
    Aka when length == vector.size[0].
    """

    segments = np.array(shaping_utils.segment_vector(vector1, 30))
    assert(segments.shape == (1, 30, 2))
    assert(np.all(segments == np.array(vector1, copy=True)))

  def __normal_large_cut():
    """
    Test segmenting for almost exact matches
    when vector1 is just a little bit shorter than length.
    """

    segments = np.array(shaping_utils.segment_vector(vector1, 28))
    assert(segments.shape == (3, 28, 2))
    assert(np.all(segments[0] == vector1[:28]))
    assert(np.all(segments[1] == vector1[1:29]))
    assert(np.all(segments[2] == vector1[2:30]))

  def __normal_short_cut():
    """
    Test for short cuts.
    """

    segments = np.array(shaping_utils.segment_vector(vector1, 5))
    assert(segments.shape == (26, 5, 2))
    assert(np.all(segments[0] == vector1[0:5]))
    assert(np.all(segments[1] == vector1[1:6]))
    assert(np.all(segments[13] == vector1[13:18]))
    assert(np.all(segments[25] == vector1[25:30]))
    assert(np.all(segments[24] == vector1[24:29]))

  __too_short()
  __exact_cut()
  __normal_large_cut()
  __normal_short_cut()


def test_shuffle_twins():
  """
  Simple test for shaping_utils.shuffle_twins(X, Y).
  """
  X = np.arange(30)
  Y = np.arange(30, 60)
  A, B = shaping_utils.shuffle_twins(X, Y)
  assert(np.where(A == 4)[0] == np.where(B == 34)[0])
  assert(np.where(A == 7)[0] == np.where(B == 37)[0])
  assert(A.shape == X.shape)
  assert(B.shape == Y.shape)

