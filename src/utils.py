import numpy as np


def get_one_hot(y):
    one_hot = np.zeros((len(y), np.max(y ) +1))
    for i, val in enumerate(list(y)):
        one_hot[i, val] = 1
    return one_hot

def split_dataset(all_data, ratio=0.75):
    n_data = len(all_data)
    training_data = all_data[0:int(0.75 * n_data)]
    evaluation_data = all_data[int(0.75 * n_data):]
    return training_data, evaluation_data