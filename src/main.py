import numpy as np
from sklearn import datasets

from src.dimensionality_reduction import DimensionalityReduction
from src.utils import get_one_hot, split_dataset
from src.multilayer_neural_network import MultiLayerNeuralNetwork
from src.simple_neural_network import SimpleNeuralNetwork
from src.loss import EntropyLoss
from src.neuralnetworkdatasets import NeuralNetworkDatasets
from visualization.plotlyvisualize import plot_decision_boundary, scatter3d_plot

dataset_name = "wine"

neural_network_datasets = NeuralNetworkDatasets()
X, Y, visualizable= neural_network_datasets.select_dataset(dataset_name)
all_data = neural_network_datasets.encode_dataset(X, Y)
training_data, evaluation_data = neural_network_datasets.split_dataset(all_data, ratio=0.75)

#multilayer neural network
first_layer_size = np.shape(X)[1]
multilayer_neural_netowrk = MultiLayerNeuralNetwork(sizes=[first_layer_size, 5, 5, 2],
                                                    loss_function=EntropyLoss,
                                                    activation_function_name="sigmoid",
                                                    n_epochs=100)
final_accuracy = multilayer_neural_netowrk.stochastic_gradient_descent(training_data=training_data,
                                                                                       evaluation_data=evaluation_data,
                                                                                       mini_batch_size=10,
                                                                                       eta=0.8,
                                                                                       lambda_=0)

print(final_accuracy)
if visualizable:
    plot_decision_boundary(lambda x: multilayer_neural_netowrk.predict(x), X, Y,
                           outputfile="./results/multi_layer_decision_boundary")

else:
    dimensionality_reduction = DimensionalityReduction(method_name="svd")
    # X, Y, Z = dimensionality_reduction.reduce_to_three_dimension()
    # scatter3d_plot(X, Y, Z,
    #                names=names,
    #                colors=labels,
    #                output_file=output_dir + "scatter3d_" + method)

    n_dim = np.shape(X)[1]
    x_mins = []
    x_maxs = []
    for i in range(n_dim):
        x_min, x_max = X[:, i].min() - .5, X[:, i].max() + .5
        x_mins.append(x_min)
        x_maxs.append(x_max)

    h = 1
    for x_min, x_max in zip(x_mins, x_maxs):
        np.arange(x_min, x_max, h)


    X_projected = dimensionality_reduction.reduce_to_two_dimension(X)
    plot_decision_boundary(lambda x: multilayer_neural_netowrk.predict(x),
                           X_projected,
                           Y,
                           outputfile="./results/multi_layer_decision_boundary")

