

class NeuralNetwork(object):

    def initialize_weights(self):
        raise NotImplementedError("override to use this method")

    def feedforward(self, X):
        raise NotImplementedError("override to use this method")

    def predict(self, X):
        raise NotImplementedError("override to use this method")

    def backwardpass(self, zs, activations, y):
        raise NotImplementedError("override to use this method")

    def stochastic_gradient_descent(self, training_data, evaluation_data, mini_batch_size, eta, lambda_):
        raise NotImplementedError("override to use this method")

    def update_mini_batch(self, mini_batch, eta):
        raise NotImplementedError("override to use this method")

    def backprob(self, X, Y):
        raise NotImplementedError("override to use this method")

    def total_loss(self, data, lambda_):
        raise NotImplementedError("override to use this method")

    def accuracy(self, data):
        raise NotImplementedError("override to use this method")