import numpy as np
from src.dimensionality_reduction import DimensionalityReduction
from src.utils import get_one_hot, split_dataset
from src.multilayer_neural_network import MultiLayerNeuralNetwork
from src.loss import EntropyLoss, L2Loss
from src.neural_network_datasets import NeuralNetworkDatasets
from visualization.plotlyvisualize import plot_decision_boundary, scatter3d_plot
from sklearn.preprocessing import scale

dataset_name = "breast_cancer"
dim_reduction = True

neural_network_datasets = NeuralNetworkDatasets()
X, Y = neural_network_datasets.select_dataset(dataset_name)
n_dimension = np.shape(X)[1]

if dim_reduction and n_dimension > 2:
    print("more than 2d case. Dimensionality reduction...")
    dimensionality_reduction = DimensionalityReduction(method_name="svd")
    X = scale(X, axis=0)
    X = dimensionality_reduction.reduce_to_two_dimension(X)
    n_dimension = np.shape(X)[1]

all_data = neural_network_datasets.encode_dataset(X, Y)
training_data, evaluation_data = neural_network_datasets.split_dataset(all_data, ratio=0.75)

#multilayer neural network
multilayer_neural_netowrk = MultiLayerNeuralNetwork(sizes=[n_dimension, 20, 10, 2],
                                                    loss_function=L2Loss,
                                                    activation_function_name="sigmoid",
                                                    n_epochs=100)
final_accuracy = multilayer_neural_netowrk.stochastic_gradient_descent(training_data=training_data,
                                                                                       evaluation_data=evaluation_data,
                                                                                       mini_batch_size=20,
                                                                                       eta=0.9,
                                                                                       lambda_=0)

print(final_accuracy)

if n_dimension <= 2 or dim_reduction:
    plot_decision_boundary(lambda x: multilayer_neural_netowrk.predict(x), X, Y,
                           outputfile="./results/multi_layer_decision_boundary_"+dataset_name)

