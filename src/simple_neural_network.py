from src.multilayer_neural_network import MultiLayerNeuralNetwork


class SimpleNeuralNetwork(MultiLayerNeuralNetwork):

    def __init__(self, sizes, loss_function, activation_function_name, n_epochs):
        if len(sizes)==3:
            super().__init__(sizes=sizes,
                             loss_function=loss_function,
                             activation_function_name=activation_function_name,
                             n_epochs=n_epochs)
        else:
            raise Exception("just accept three layer neural network")


    def predict(self, X):
        return super().predict(X=X)

    def backwardpass(self, zs, activations, y):
        return super().backwardpass(zs=zs, activations=activations, y=y)

    def stochastic_gradient_descent(self, training_data, evaluation_data, mini_batch_size, eta, lambda_):
        return super().stochastic_gradient_descent(training_data=training_data,
                                            evaluation_data=evaluation_data,
                                            mini_batch_size=mini_batch_size,
                                            eta=eta,
                                            lambda_=lambda_)

    def update_mini_batch(self, mini_batch, eta):
        return super().update_mini_batch(mini_batch=mini_batch,
                                  eta=eta)

    def backprob(self, X, Y):
        return super().backprob(X=X, Y=Y)


    # def total_loss(self, data, lambda_):
    #     super().total_loss(data=data, lambda_=lambda_)

    def accuracy(self, data):
        return super().accuracy(data=data)

    def initialize_weights(self, method="xavier"):
        return super().initialize_weights(method=method)

    def feedforward(self, X):
        return super().feedforward(X=X)
