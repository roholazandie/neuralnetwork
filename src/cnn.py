import tensorflow
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from src.neural_network_datasets import MNISTDataset
import matplotlib.pyplot as plt
from keras import losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras import backend as K
import numpy as np
import os

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


architechure = "model2" #"simple", "model2"
num_classes = 10
n_epochs = 10
batch_size = 50

# Center
centered_data_dir = "../data/centered_data/"
centered_mnist_dataset = MNISTDataset(dataset_dir= centered_data_dir)

centered_train_data, centered_train_labels = centered_mnist_dataset.load_train_dataset()
centered_test_data, centered_test_labels = centered_mnist_dataset.load_test_dataset()
centered_validation_data, centered_validation_labels = centered_mnist_dataset.load_validation_dataset()

centered_train_data = centered_mnist_dataset.reshape_dataset(centered_train_data)
centered_test_data = centered_mnist_dataset.reshape_dataset(centered_test_data)
centered_validation_data = centered_mnist_dataset.reshape_dataset(centered_validation_data)

one_hotted_centered_train_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_train_labels)
one_hotted_centered_test_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_test_labels)
one_hotted_centered_validation_labels = centered_mnist_dataset.one_hot_encode_dataset(centered_validation_labels)

# UnCenter
uncentered_data_dir = "../data/uncentered_data/"
uncentered_mnist_dataset = MNISTDataset(dataset_dir= uncentered_data_dir)

uncentered_train_data, uncentered_train_labels = uncentered_mnist_dataset.load_train_dataset()
uncentered_test_data, uncentered_test_labels = uncentered_mnist_dataset.load_test_dataset()
uncentered_validation_data, uncentered_validation_labels = uncentered_mnist_dataset.load_validation_dataset()

uncentered_train_data = uncentered_mnist_dataset.reshape_dataset(uncentered_train_data)
uncentered_test_data = uncentered_mnist_dataset.reshape_dataset(uncentered_test_data)
uncentered_validation_data = uncentered_mnist_dataset.reshape_dataset(uncentered_validation_data)


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

# Position and Size Invariance dataset
psi_data_dir = "../data/position_and_size_invariance/"
psi_mnist_dataset = MNISTDataset(dataset_dir= psi_data_dir)

psi_train_data, psi_train_labels = psi_mnist_dataset.load_train_dataset()
psi_validation_data, psi_validation_labels = psi_mnist_dataset.load_test_dataset()
# psi_validation_data, psi_validation_labels = psi_mnist_dataset.load_validation_dataset()


# for i in range(0, 9):
# 	plt.subplot(331 + i)
# 	plt.imshow(psi_train_data[i+20][0].reshape(28,28), cmap=plt.get_cmap('gray'))
# plt.show()

psi_train_data = psi_mnist_dataset.reshape_dataset(psi_train_data)
psi_validation_data = psi_mnist_dataset.reshape_dataset(psi_validation_data)
# psi_validation_data = psi_mnist_dataset.reshape_dataset(psi_validation_data)

one_hotted_psi_train_labels = psi_mnist_dataset.one_hot_encode_dataset(psi_train_labels)
one_hotted_psi_validation_labels = psi_mnist_dataset.one_hot_encode_dataset(psi_validation_labels)
# one_hotted_psi_validation_labels = psi_mnist_dataset.one_hot_encode_dataset(psi_validation_labels)




# Model Architecture
if architechure =="simple":
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=[metrics.categorical_accuracy])

elif architechure == "model1":
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=[metrics.categorical_accuracy])

elif architechure == "model2":
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    n_epochs = 30
    batch_size = 86

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        # zoom_range=0.1,  # Randomly zoom image
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        # horizontal_flip=False,  # randomly flip images
        # vertical_flip=False
        )
    datagen.fit(psi_train_data)


# # Center Model
# tensorboard = TensorBoard(log_dir="logs_cnn_centered")
# model.fit(centered_train_data, one_hotted_centered_train_labels,
#           batch_size=batch_size,
#           epochs=n_epochs,
#           verbose=1,
#           validation_data=(centered_validation_data, one_hotted_centered_validation_labels),
#           callbacks=[tensorboard])
# score = model.evaluate(centered_test_data, one_hotted_centered_test_labels, verbose=0)
# print('loss:', score[0], 'categorical_accuracy:', score[1])
# # tensorboard --logdir=foo:C:\Users\Mohsen.SharifiRenani\Desktop\pycharmprojects\neuralnetwork3\logs_cnn_centered
#
# # Uncentered Model
# tensorboard = TensorBoard(log_dir="logs_cnn_uncentered")
# model.fit(centered_train_data, one_hotted_centered_train_labels,
#           batch_size=batch_size,
#           epochs=n_epochs,
#           verbose=1,
#           validation_data=(uncentered_validation_data, one_hotted_uncentered_validation_labels),
#           callbacks=[tensorboard])
# score = model.evaluate(uncentered_test_data, one_hotted_uncentered_test_labels, verbose=0)
# print('loss:', score[0], 'categorical_accuracy:', score[1])
#
# # tensorboard --logdir=foo:C:\Users\Mohsen.SharifiRenani\Desktop\pycharmprojects\neuralnetwork3\logs_cnn_uncentered
#
# # Center_Uncentered Model
#
# tensorboard = TensorBoard(log_dir="logs_cnn_centered_uncentered")
# model.fit(centered_uncentered_train_data, one_hotted_centered_uncentered_train_labels,
#           batch_size=batch_size,
#           epochs=n_epochs,
#           verbose=1,
#           validation_data=(centered_uncentered_validation_data, one_hotted_centered_uncentered_validation_labels),
#           callbacks=[tensorboard])
# score = model.evaluate(centered_uncentered_test_data, one_hotted_centered_uncentered_test_labels, verbose=0)
# print('loss:', score[0], 'categorical_accuracy:', score[1])

# tensorboard --logdir=foo:C:\Users\Mohsen.SharifiRenani\Desktop\pycharmprojects\neuralnetwork3\logs_cnn_centered_uncentered


#Position invariance
tensorboard = TensorBoard(log_dir="../logs/logs_cnn_psi")
# model.fit(psi_train_data, one_hotted_psi_train_labels,
#           batch_size=batch_size,
#           epochs=n_epochs,
#           verbose=1,
#           validation_data=(psi_validation_data, one_hotted_psi_validation_labels),
#           callbacks=[tensorboard])


history = model.fit_generator(datagen.flow(psi_train_data,
                      one_hotted_psi_train_labels,
                      batch_size = batch_size),
                      epochs = n_epochs,
                      validation_data = (psi_validation_data, one_hotted_psi_validation_labels),
                      verbose = 2,
                      steps_per_epoch=psi_train_data.shape[0] // batch_size,
                      callbacks=[learning_rate_reduction, tensorboard])

#score = model.evaluate(centered_uncentered_test_data, one_hotted_centered_uncentered_test_labels, verbose=0)
#print('loss:', score[0], 'categorical_accuracy:', score[1])



json_string = model.to_json()
name = "rohola_mohsen"
open(name + '_architecture.json','w').write(json_string)

model.save_weights(name + '_weights.h5')