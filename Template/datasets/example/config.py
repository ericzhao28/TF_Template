'''
Configuration for example dataset.
'''

import os


DUMPS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

EXAMPLE_FEATURES_PATH = DUMPS_DIR + "example_X.np"
EXAMPLE_LABELS_PATH = DUMPS_DIR + "example_Y.np"
n_steps = 8

