import os


SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saves/"
GRAPHS_TRAIN_DIR = SAVES_DIR + "graphs/train/"
GRAPHS_TEST_DIR = SAVES_DIR + "graphs/test/"
CHECKPOINTS_DIR = SAVES_DIR + "checkpoints/"

ITERATIONS = 32
BATCH_SIZE = 20
N_FEATURES = 4
N_STEPS = 10
LAYERS = {
    'h_gru': 10,
    'h_att': 10,
}
ENCODED_DIM = 10
N_CLASSES = 10

