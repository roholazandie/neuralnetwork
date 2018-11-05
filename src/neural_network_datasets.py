import numpy as np
from sklearn import datasets
from src.utils import get_one_hot
from sklearn.preprocessing import scale
import pickle
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


class MNISTDataset():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def load_train_dataset(self):
        with open(self.dataset_dir + "Train.pickle", 'rb') as file_reader:
            centered_train_data = pickle.load(file_reader, encoding="latin1")


        with open(self.dataset_dir + "Train_Lbs.pickle", 'rb') as file_reader:
            centered_train_labels = pickle.load(file_reader, encoding="latin1")

        return centered_train_data, centered_train_labels


    def load_test_dataset(self):
        with open(self.dataset_dir + "Test.pickle", 'rb') as file_reader:
            centered_test_data = pickle.load(file_reader, encoding="latin1")

        with open(self.dataset_dir + "Test_Lbs.pickle", 'rb') as file_reader:
            centered_test_labels = pickle.load(file_reader, encoding="latin1")

        return centered_test_data, centered_test_labels

    def load_validation_dataset(self):
        with open(self.dataset_dir + "Validation.pickle", 'rb') as file_reader:
            centered_validation_data = pickle.load(file_reader, encoding="latin1")

        with open(self.dataset_dir + "Validation_Lbs.pickle", 'rb') as file_reader:
            centered_validation_labels = pickle.load(file_reader, encoding="latin1")

        return centered_validation_data, centered_validation_labels


    # def scale(self, dataset):
    #     scaled_dataset = scale(dataset, axis=0)
    #     return scaled_dataset

    def one_hot_encode_dataset(self, Y):
        Y_one_hot = get_one_hot(Y)
        return Y_one_hot

    def reshape_dataset(self, dataset):
        return dataset.reshape(np.shape(dataset)[0], 28, 28, 1).astype('float32')



if __name__ == "__main__":
    nndataset = NeuralNetworkDatasets()
    nndataset.select_dataset(dataset_name="wine")