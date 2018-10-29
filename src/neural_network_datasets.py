import numpy as np
from sklearn import datasets
from src.utils import get_one_hot
import random

class NeuralNetworkDatasets():

    def __init__(self):
        pass

    def select_dataset(self, dataset_name="two_moons"):
        np.random.seed(0)
        X, Y = [], []
        if dataset_name == "two_moons":
            X, Y = datasets.make_moons(200, noise=0.25)

        elif dataset_name == "iris":
            X, Y = datasets.load_iris(return_X_y=True)
            Y = [0 if y == 0 else 1 for y in Y]

        elif dataset_name == "wine":
            X, Y = datasets.load_wine(return_X_y=True)
            Y = [0 if y != 0 else 1 for y in Y]

        elif dataset_name == "breast_cancer":
            X, Y = datasets.load_breast_cancer(return_X_y=True)

        return X, Y,

    def encode_dataset(self, X, Y):
        Y_one_hot = get_one_hot(Y)
        all_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X, Y_one_hot)]
        return all_data


    def split_dataset(self, all_data, ratio = 0.75):
        n_data = len(all_data)
        random.shuffle(all_data)
        training_data = all_data[0:int(ratio * n_data)]
        evaluation_data = all_data[int(ratio * n_data):]
        return training_data, evaluation_data


if __name__ == "__main__":
    nndataset = NeuralNetworkDatasets()
    nndataset.select_dataset(dataset_name="wine")