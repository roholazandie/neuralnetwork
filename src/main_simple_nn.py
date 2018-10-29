from src.dimensionality_reduction import DimensionalityReduction
from src.neural_network_datasets import NeuralNetworkDatasets
from src.simple_neural_network import SimpleNeuralNetwork
from src.loss import EntropyLoss, L2Loss
from sklearn.preprocessing import scale
import numpy as np

#Three layer neural network
from visualization.plotlyvisualize import plot_decision_boundary

dataset_name = "breast_cancer"

neural_network_datasets = NeuralNetworkDatasets()
X, Y = neural_network_datasets.select_dataset(dataset_name)
n_dimension = np.shape(X)[1]
if n_dimension> 2:
    print("more than 2d case")
    dimensionality_reduction = DimensionalityReduction(method_name="svd")
    X = scale(X, axis=0)
    X = dimensionality_reduction.reduce_to_two_dimension(X)
    n_dimension = np.shape(X)[1]

all_data = neural_network_datasets.encode_dataset(X, Y)
training_data, evaluation_data = neural_network_datasets.split_dataset(all_data, ratio=0.75)


simple_neural_network = SimpleNeuralNetwork(sizes=[2, 5, 2],
                                            loss_function=EntropyLoss,
                                            activation_function_name="sigmoid",
                                            n_epochs=100)

final_accuracy = simple_neural_network.stochastic_gradient_descent(training_data=all_data,
                                                                   evaluation_data=evaluation_data,
                                                                   mini_batch_size=10,
                                                                   eta=0.8,
                                                                   lambda_=0)

plot_decision_boundary(lambda x: simple_neural_network.predict(x), X, Y,
                       outputfile="../results/three_layer_decision_boundary_simple"+dataset_name)