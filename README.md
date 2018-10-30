## Neural network from scratch
This repo contains the code for three layer(simple neural network) and mutilayer neural networks with numpy.

### Datasets
We use different toy datasets in sklearn like iris, wine, two moons and breast cancer.In cases that we have more than one class we just use one class as 0 and others as 1.

### Visualization
For visualization we use plotly and 2d scatter. For datasets with more than 2 dimension we use dimensionality reduction technique known as SVD to choose two highest PCA and map all dataset two 2d plane and feed to network.
it is also possible to feed more than dimension to neural network without visualization.

### Installation

```
pip install -r requirements.txt
```