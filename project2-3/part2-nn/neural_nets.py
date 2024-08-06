import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    if x > 0:
        return x
    else:
        return 0

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if x > 0:
        return 1
    else:
        return 0

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):
        v_activate = np.vectorize(rectified_linear_unit)
        v_relu_derivative = np.vectorize(rectified_linear_unit_derivative)

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        # z = w*x + b
        hidden_layer_weighted_input = \
            self.input_to_hidden_weights @ input_values + self.biases # 2 by 1
        # f(z)
        hidden_layer_activation = v_activate(hidden_layer_weighted_input) # 3 by 1

        # u = sum(v*f(x))
        output = (self.hidden_to_output_weights @ hidden_layer_activation).item()
        # a = f(u)
        activated_output = output_layer_activation(output)

        lost = (y - activated_output) ** 2 / 2

        ### Backpropagation ###

        # Compute gradients
        output_layer_error = -(y - activated_output)
        hidden_layer_error = output_layer_error * self.hidden_to_output_weights.T # 3 by 1 matrix

        # dC/dv =   dC/do  * do/du *        du/dv
        # dC/dv = -(y - a) *   1   * [f(z1), f(z2), f(z3)])
        dodu = output_layer_activation_derivative(output) # Not used (evaluated as 1)
        hidden_to_output_weight_gradients = \
            np.multiply(output_layer_error, hidden_layer_activation).T

        # dC/dw =   dC/do  * do/du *     du/df    *   df/dz     *  dz/dw
        # dC/dw = -(y - a) *   1   * [v1, v2, v3] * dReLu(f)/dz * [x1,x2]
        dfdz = v_relu_derivative(hidden_layer_weighted_input)
        dCdo_dodu_dudf_dfdz = np.multiply(hidden_layer_error, dfdz)
        input_to_hidden_weight_gradients = dCdo_dodu_dudf_dfdz @ input_values.T 

        # dC/db =   dC/do  * do/du *     du/df    *   df/dz     * dz/db
        # dC/db = -(y - a) *   1   * [v1, v2, v3] * dReLu(f)/dz *   1
        bias_gradients = dCdo_dodu_dudf_dfdz 

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate * bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - \
                                       self.learning_rate * \
                                       input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - \
                                        self.learning_rate * \
                                        hidden_to_output_weight_gradients 

        print("u = {}".format(output))
        print("a = {}".format(activated_output))
        print("Grad_V = {}".format(self.hidden_to_output_weights))
        print("Grad_W = {}".format(input_to_hidden_weight_gradients))
        print("Grad_b = {}".format(bias_gradients))
        print("V = {}".format(self.hidden_to_output_weights))
        print("W = {}".format(self.input_to_hidden_weights))
        print("b = {}".format(self.biases))
        print("lost = {}".format(lost))
        print("-----")

    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        v_activate = np.vectorize(rectified_linear_unit)

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = \
            self.input_to_hidden_weights @ input_values + self.biases # 3 by 1 
        hidden_layer_activation = v_activate(hidden_layer_weighted_input) # 3 by 1

        output = self.hidden_to_output_weights @ hidden_layer_activation
        activated_output = output_layer_activation(output)

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
