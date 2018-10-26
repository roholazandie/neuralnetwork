import numpy as np
import random


class MultiLayerNeuralNetwork():

    def __init__(self, sizes, loss_function, activation_function_name, n_epochs):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.loss = loss_function
        self.activation_function_name = activation_function_name
        if activation_function_name == "sigmoid":
            self.activation_function = lambda x: self.sigmoid(x)
        elif activation_function_name == "tanh":
            self.activation_function = lambda x: self.tanh(x)

        self.biases = []
        self.weights = []
        self.n_epochs = n_epochs

        self.initialize_weights(method="normal")

    @property
    def grad_b(self):
        return self._grads_b

    @property
    def grad_w(self):
        return self._grads_w

    def grad_activation_function(self, z):
        if self.activation_function_name == "sigmoid":
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.activation_function_name == "tanh":
            return 1 - self.tanh(z) ** 2

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def initialize_weights(self, method="xavier"):
        if method == "xavier":
            self.biases = [np.random.randn(size, 1) for size in self.sizes[1:]]
            self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        elif method == "normal":
            self.biases = [np.random.randn(size, 1) for size in self.sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, X):
        for bias, weight in zip(self.biases, self.weights):
            X = self.activation_function(np.dot(weight, X) + bias)
        return X

    def _feedforward(self, X):
        activation = X
        activations = [X]
        zs = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            activation = self.activation_function(z)
            zs.append(z)
            activations.append(activation)

        return zs, activations

    def predict(self, X):
        X = np.transpose(X)
        for bias, weight in zip(self.biases, self.weights):
            X = self.activation_function(np.dot(weight, X) + bias)
        predictions = np.argmax(X, axis=0)
        return predictions


    def backwardpass(self, zs, activations, y):
        grad_z = self.grad_activation_function(zs[-1])
        delta = self.loss.delta(grad_z, activations[-1], y)
        # delta = self.loss.delta(zs[-1], activations[-1], y)
        self._grads_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        self._grads_b[-1] = delta

        # this is a countdown for loop
        for l in range(self.num_layers - 2, 0, -1):
            z = zs[l - 1]
            delta = np.dot(np.transpose(self.weights[l]), delta) * self.grad_activation_function(z)
            self._grads_w[l - 1] = np.dot(delta, np.transpose(activations[l - 1]))
            self._grads_b[l - 1] = delta

        # #this is a regular for loop
        # for l in range(2, self.num_layers):
        #     z = zs[-l]
        #     delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.grad_activation_function(z)
        #     self._grads_b[-l] = delta
        #     self._grads_w[-l] = np.dot(delta, activations[-l - 1].transpose())

    def stochastic_gradient_descent(self, training_data, evaluation_data, mini_batch_size, eta, lambda_):
        n = len(training_data)
        train_accuracies = []
        animate = True
        animation_data = []
        monitor_accuracy = False
        monitor_loss = False
        evaluate = True

        for i in range(self.n_epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            loss = self.total_loss(training_data, lambda_)
            if monitor_loss:
                print(loss)

            train_accuracy = self.accuracy(training_data)
            train_accuracies.append(train_accuracy)
            if monitor_accuracy:
                print(train_accuracy)

            if evaluate:
                evaluation_accuracy = self.accuracy(evaluation_data)
                print(evaluation_accuracy)

        final_accuracy = np.mean(train_accuracies)
        return final_accuracy, animation_data

    def predict_for_space(self, X):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        return Z

    def update_mini_batch(self, mini_batch, eta):
        total_batch_grads_w = [np.zeros(np.shape(weight)) for weight in self.weights]
        total_batch_grads_b = [np.zeros(np.shape(bias)) for bias in self.biases]

        for X, Y in mini_batch:
            self.backprob(X, Y)
            for i, (grad_w, grad_b) in enumerate(zip(self._grads_w, self._grads_b)):
                total_batch_grads_w[i] += grad_w
                total_batch_grads_b[i] += grad_b

        self.weights = [weight - (eta / len(mini_batch)) * total_batch_grad_w for weight, total_batch_grad_w in
                        zip(self.weights, total_batch_grads_w)]
        self.biases = [bias - (eta / len(mini_batch)) * total_batch_grad_b for bias, total_batch_grad_b in
                       zip(self.biases, total_batch_grads_b)]

    def backprob(self, X, Y):
        self._grads_w = [np.zeros(np.shape(weight)) for weight in self.weights]
        self._grads_b = [np.zeros(np.shape(bias)) for bias in self.biases]

        zs, activations = self._feedforward(X)
        self.backwardpass(zs, activations, Y)

    def total_loss(self, data, lambda_):
        loss = 0.0
        for x, y in data:
            a = self.feedforward(x)
            loss += self.loss.calculate(a, y) / len(data)
        loss += 0.5 * (lambda_ / len(data)) * sum(np.linalg.norm(weight) ** 2 for weight in self.weights)
        return loss

    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        return np.mean([float(x == y) for (x, y) in results])