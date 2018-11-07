import tensorflow as tf
from keras.callbacks import TensorBoard
from src.neural_network_datasets import MNISTDataset
import matplotlib.pyplot as plt
from keras import losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


input_shape = (784,)

num_classes = 10
n_epochs = 10
batch_size = 50

# Center
centered_data_dir = "R:/Research Common/EBL/Researcher/Mohsen/Class/Fall 2018/Neural Network/project02/data/centered_data/"
centered_mnist_dataset = MNISTDataset(dataset_dir= centered_data_dir)

centered_train_data, centered_train_labels = centered_mnist_dataset.load_train_dataset()
centered_test_data, centered_test_labels = centered_mnist_dataset.load_test_dataset()
centered_validation_data, centered_validation_labels = centered_mnist_dataset.load_validation_dataset()

one_hotted_centered_train_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_train_labels)
one_hotted_centered_test_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_test_labels)
one_hotted_centered_validation_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_validation_labels)

# UN-Center
uncentered_data_dir = "R:/Research Common/EBL/Researcher/Mohsen/Class/Fall 2018/Neural Network/project02/data/uncentered_data/"
uncentered_mnist_dataset = MNISTDataset(dataset_dir= uncentered_data_dir)

uncentered_train_data, uncentered_train_labels = uncentered_mnist_dataset.load_train_dataset()
uncentered_test_data, uncentered_test_labels = uncentered_mnist_dataset.load_test_dataset()
uncentered_validation_data, uncentered_validation_labels = uncentered_mnist_dataset.load_validation_dataset()

uncentered_train_data = np.squeeze(uncentered_train_data)
uncentered_test_data = np.squeeze(uncentered_test_data)
uncentered_validation_data = np.squeeze(uncentered_validation_data)

one_hotted_uncentered_train_labels = uncentered_mnist_dataset.one_hot_encode_dataset(uncentered_train_labels)
one_hotted_uncentered_test_labels = uncentered_mnist_dataset.one_hot_encode_dataset(uncentered_test_labels)
one_hotted_uncentered_validation_labels = uncentered_mnist_dataset.one_hot_encode_dataset(uncentered_validation_labels)

# combine Center-UnCenter
centered_uncentered_train_data = np.vstack((centered_train_data, uncentered_train_data))
centered_uncentered_test_data = np.vstack((centered_test_data, uncentered_test_data))
centered_uncentered_validation_data = np.vstack((centered_validation_data, uncentered_validation_data))

one_hotted_centered_uncentered_train_labels = np.vstack((one_hotted_centered_train_labels, one_hotted_uncentered_train_labels))
one_hotted_centered_uncentered_test_labels = np.vstack((one_hotted_centered_test_labels, one_hotted_uncentered_test_labels))
one_hotted_centered_uncentered_validation_labels = np.vstack((one_hotted_centered_validation_labels, one_hotted_uncentered_validation_labels))

shuffle_index_train = np.arange(centered_uncentered_train_data.shape[0])
shuffle_index_test = np.arange(centered_uncentered_test_data.shape[0])
shuffle_index_validation = np.arange(centered_uncentered_validation_data.shape[0])

np.random.shuffle(shuffle_index_train)
np.random.shuffle(shuffle_index_test)
np.random.shuffle(shuffle_index_validation)

centered_uncentered_train_data = centered_uncentered_train_data[shuffle_index_train]
centered_uncentered_test_data = centered_uncentered_test_data[shuffle_index_test]
centered_uncentered_validation_data = centered_uncentered_validation_data[shuffle_index_validation]

one_hotted_centered_uncentered_train_labels = one_hotted_centered_uncentered_train_labels[shuffle_index_train]
one_hotted_centered_uncentered_test_labels = one_hotted_centered_uncentered_test_labels[shuffle_index_test]
one_hotted_centered_uncentered_validation_labels = one_hotted_centered_uncentered_validation_labels[shuffle_index_validation]

# Position and Size Invariance
psi_data_dir = "R:/Research Common/EBL/Researcher/Mohsen/Class/Fall 2018/Neural Network/project02/data/position_and_size_invariance/"
psi_mnist_dataset = MNISTDataset(dataset_dir= psi_data_dir)

psi_train_data, psi_train_labels = psi_mnist_dataset.load_train_dataset()
psi_test_data, psi_test_labels = psi_mnist_dataset.load_test_dataset()
# psi_validation_data, psi_validation_labels = psi_mnist_dataset.load_validation_dataset()

psi_train_data = np.squeeze(psi_train_data)
psi_test_data = np.squeeze(psi_test_data)
# psi_validation_data = np.squeeze(psi_validation_data)

one_hotted_psi_train_labels = psi_mnist_dataset.one_hot_encode_dataset(psi_train_labels)
one_hotted_psi_test_labels = psi_mnist_dataset.one_hot_encode_dataset(psi_test_labels)
# one_hotted_psi_validation_labels = psi_mnist_dataset.one_hot_encode_dataset(psi_validation_labels)
############################################
# main model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=input_shape))
model.add(Dense(10, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adadelta(),
              metrics=[metrics.categorical_accuracy, metrics.mean_squared_error])

# Center model
tensorboard = TensorBoard(log_dir="logs_three_layer_centered")
model.fit(centered_train_data, one_hotted_centered_train_labels,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=(centered_validation_data, one_hotted_centered_validation_labels),
          callbacks=[tensorboard])
score = model.evaluate(centered_test_data, one_hotted_centered_test_labels, verbose=0)
print('loss:', score[0], 'categorical_accuracy:', score[1])
# tensorboard --logdir=foo:C:\Users\Mohsen.SharifiRenani\Desktop\pycharmprojects\neuralnetwork3\logs_three_layer_centered

# Un Centered Model
tensorboard = TensorBoard(log_dir="logs_three_layer_uncentered")
model.fit(centered_train_data, one_hotted_centered_train_labels,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=(uncentered_validation_data, one_hotted_uncentered_validation_labels),
          callbacks=[tensorboard])
score = model.evaluate(uncentered_test_data, one_hotted_uncentered_test_labels, verbose=0)
print('loss:', score[0], 'categorical_accuracy:', score[1])

# tensorboard --logdir=foo:C:\Users\Mohsen.SharifiRenani\Desktop\pycharmprojects\neuralnetwork3\logs_three_layer_uncentered

# Centered_Un Centered Model
tensorboard = TensorBoard(log_dir="logs_three_layer_center_uncentered")
model.fit(centered_uncentered_train_data, one_hotted_centered_uncentered_train_labels,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=(centered_uncentered_validation_data, one_hotted_centered_uncentered_validation_labels),
          callbacks=[tensorboard])
score = model.evaluate(centered_uncentered_test_data, one_hotted_centered_uncentered_test_labels, verbose=0)
print('loss:', score[0], 'categorical_accuracy:', score[1])

# tensorboard --logdir=foo:C:\Users\Mohsen.SharifiRenani\Desktop\pycharmprojects\neuralnetwork3\logs_three_layer_center_uncentered
