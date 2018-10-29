#Three layer neural network


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
                       outputfile="./results/three_layer_decision_boundary")