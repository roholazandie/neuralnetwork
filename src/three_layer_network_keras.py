from keras.callbacks import TensorBoard
from src.neural_network_datasets import MNISTDataset
import matplotlib.pyplot as plt
from keras import losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

input_shape = (784,)

num_classes = 10
n_epochs = 50
batch_size = 10

centered_data_dir = "../data/centered_data/"
centered_mnist_dataset = MNISTDataset(dataset_dir= centered_data_dir)

centered_train_data, centered_train_labels = centered_mnist_dataset.load_train_dataset()
centered_test_data, centered_test_labels = centered_mnist_dataset.load_test_dataset()
centered_validation_data, centered_validation_labels = centered_mnist_dataset.load_validation_dataset()

# centered_train_data = centered_train_data.reshape(np.shape(centered_train_data)[0], 784, 1).astype('float32')
# centered_test_data = centered_test_data.reshape(np.shape(centered_test_data)[0], 784, 1).astype('float32')
# centered_validation_data = centered_validation_data.reshape(np.shape(centered_validation_data)[0], 784, 1).astype('float32')

one_hotted_centered_train_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_train_labels)
one_hotted_centered_test_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_test_labels)
one_hotted_centered_validation_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_validation_labels)


tensorboard = TensorBoard(log_dir="logs_three_layer_centered")

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=input_shape))
model.add(Dense(10, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adadelta(),
              metrics=[metrics.categorical_accuracy, metrics.mean_squared_error])

model.fit(centered_train_data, one_hotted_centered_train_labels,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=(centered_validation_data, one_hotted_centered_validation_labels),
          callbacks=[tensorboard])
score = model.evaluate(centered_test_data, one_hotted_centered_test_labels, verbose=0)