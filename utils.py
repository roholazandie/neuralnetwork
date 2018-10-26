import numpy as np


def get_one_hot(y):
    one_hot = np.zeros((len(y), np.max(y ) +1))
    for i, val in enumerate(list(y)):
        one_hot[i, val] = 1
    return one_hot