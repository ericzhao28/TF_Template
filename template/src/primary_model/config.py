import os

ITERATIONS = 23
BATCH_SIZE = 3
EMB_DIM = 300
EMB_PATH = os.path.dirname(os.path.realpath(__file__)) + "/embeddings.vector"
N_STEPS = 7
LAYERS = {
    'H_GRU': 11,
    'H_ATT': 13,
}
ENCODED_DIM = 17
N_CLASSES = 2
