import numpy as np
from sklearn import datasets
from utils import get_one_hot
from multilayer_neural_network import MultiLayerNeuralNetwork
from loss import EntropyLoss, L2Loss
from visualization.plotlyvisualize import plot_decision_boundary

np.random.seed(0)
X, Y = datasets.make_moons(200, noise=0.25)
Y_one_hot = get_one_hot(Y)
all_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(X, Y_one_hot)]
n_data = len(all_data)
training_data = all_data[0:int(0.75 * n_data)]
evaluation_data = all_data[int(0.75 * n_data):]

multilayer_neural_netowrk = MultiLayerNeuralNetwork(sizes=[2, 5, 5, 2],
                                                    loss_function=EntropyLoss,
                                                    activation_function_name="sigmoid",
                                                    n_epochs=100)
final_accuracy, animation_data = multilayer_neural_netowrk.stochastic_gradient_descent(training_data=all_data,
                                                                                       evaluation_data=evaluation_data,
                                                                                       mini_batch_size=10,
                                                                                       eta=0.8,
                                                                                       lambda_=0)

plot_decision_boundary(lambda x: multilayer_neural_netowrk.predict(x), X, Y)