from src.neural_network_datasets import MNISTDataset
import matplotlib.pyplot as plt
from keras import losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import backend as K

num_classes = 10
n_epochs = 3
batch_size = 100

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


#Visualize the first 9 samples
# for i in range(0,9):
# 	plt.subplot(331 + i)
# 	plt.imshow(centered_train_data[i][0], cmap=plt.get_cmap('gray'))
# plt.show()



#################model
tensorboard = TensorBoard(log_dir="logs_cnn_centered")

input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adadelta(),
              metrics=[metrics.categorical_accuracy])



model.fit(centered_train_data, one_hotted_centered_train_labels,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=(centered_validation_data, one_hotted_centered_validation_labels),
          callbacks=[tensorboard])
score = model.evaluate(centered_test_data, one_hotted_centered_test_labels, verbose=0)