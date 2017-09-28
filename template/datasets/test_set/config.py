'''
Standard configuration for primary_set dataset.
'''

import os


DUMPS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"
SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/"
GRAPHS_DIR = SAVES_DIR + "graphs/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

DOWNLOAD_URL = "https://clubpenguin.com"
RAW_SAVE_NAME = "raw_dataset.csv"
PROCESSED_SAVE_NAME = "processed_dataset.p"

SEQ_LEN = 10

numerical_fields = ["small_relevant", "small_random", "big_random", "big_relevant"]
label_field = "label"
seq_field = "seq_number"

label_classes = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90"]

