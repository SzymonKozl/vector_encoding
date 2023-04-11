import gensim
import json
import os
import numpy as np

with open(os.path.join(os.path.dirname(__file__), "load_config.json"), "r") as f:
    config = json.load(f)

DATA_PATH = config["data_path"]


kv = None


def load() -> np.ndarray:
    global kv
    if not kv:
        kv = gensim.models.KeyedVectors.load_word2vec_format(DATA_PATH, binary=DATA_PATH.endswith('.bin'))
    return kv.vectors
