from src.neural_network_datasets import MNISTDataset
import numpy as np
from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding

from visualization.plotlyvisualize import scatter_plot, scatter3d_plot

output_dir = "../output/"
dataset_name = "psi"

if dataset_name == "centered":
    centered_data_dir = "../data/centered_data/"
    centered_mnist_dataset = MNISTDataset(dataset_dir= centered_data_dir)
    centered_test_data, centered_test_labels = centered_mnist_dataset.load_test_dataset()

    centered_test_data = centered_test_data[0:1000,:]
    centered_test_labels = centered_test_labels[0:1000]

    #centered_train_data_transformed = TSNE(n_components = 3, perplexity=40, verbose=2).fit_transform(centered_test_data)

    centered_train_data_transformed = SpectralEmbedding(n_neighbors=10, n_components=3).fit_transform(centered_test_data)

    scatter3d_plot(centered_train_data_transformed[:, 0],
                   centered_train_data_transformed[:, 1],
                   centered_train_data_transformed[:, 2],
                   names=centered_test_labels,
                   colors=centered_test_labels,
                   output_file=output_dir + "scatter3d_lle")



if dataset_name == "uncentered":
    uncentered_data_dir = "../data/uncentered_data/"
    uncentered_mnist_dataset = MNISTDataset(dataset_dir= uncentered_data_dir)
    uncentered_test_data, uncentered_test_labels = uncentered_mnist_dataset.load_test_dataset()
    uncentered_test_data = np.squeeze(uncentered_test_data)

    #uncentered_test_data = uncentered_test_data[0:1000,:]
    #uncentered_test_labels = uncentered_test_labels[0:1000]

    transformer = TSNE(n_components = 3, perplexity=40, verbose=2)
    uncentered_train_data_transformed = transformer.fit_transform(uncentered_test_data)


    scatter3d_plot(uncentered_train_data_transformed[:, 0],
                   uncentered_train_data_transformed[:, 1],
                   uncentered_train_data_transformed[:, 2],
                   names=uncentered_test_labels,
                   colors=uncentered_test_labels,
                   output_file=output_dir + "scatter3d_uncentered")


if dataset_name == "psi":
    psi_data_dir = "../data/position_and_size_invariance/"
    psi_mnist_dataset = MNISTDataset(dataset_dir= psi_data_dir)
    psi_test_data, psi_test_labels = psi_mnist_dataset.load_test_dataset()
    psi_test_data = np.squeeze(psi_test_data)

    psi_test_data = psi_test_data[0:1000,:]
    psi_test_labels = psi_test_labels[0:1000]

    #transformer = TSNE(n_components = 3, perplexity=40, verbose=2)
    #psi_train_data_transformed = transformer.fit_transform(psi_test_data)

    psi_train_data_transformed = SpectralEmbedding(n_neighbors=10, n_components=3).fit_transform(psi_test_data)

    scatter3d_plot(psi_train_data_transformed[:, 0],
                   psi_train_data_transformed[:, 1],
                   psi_train_data_transformed[:, 2],
                   names=psi_test_labels,
                   colors=psi_test_labels,
                   output_file=output_dir + "scatter3d_psi_spectral")
