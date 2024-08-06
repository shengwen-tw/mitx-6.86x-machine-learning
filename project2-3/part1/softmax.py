import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    H = np.zeros([theta.shape[0], X.shape[0]])
    e_theta_j = 0

    # Iterate through all features
    for i in range(0, X.shape[0]):
        x = X[[i], :]
        sum = 0

        # Calculate max_j(theta_j * x / tau)
        c = np.dot(theta[[0], :], x.T) / temp_parameter
        for j in range(1, theta.shape[0]):
            c_new = np.dot(theta[[j], :], x.T) / temp_parameter
            if c_new > c:
                c = c_new

        # Iterate through all kind of labels
        for j in range(0, theta.shape[0]):
            e_theta_j = np.exp(np.dot(theta[[j], :], x.T) / temp_parameter - c)
            H[j, i] = e_theta_j
            sum = sum + e_theta_j

        # Normalization
        for j in range(0, theta.shape[0]):
            H[j, i] = H[j, i] / sum

    return H

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    penality = 0
    for i in range(0, X.shape[0]):
        x = X[[i], :]
        xt = x.T

        # Calculate max_j(theta_j * x / tau)
        c = np.dot(theta[[0], :], xt) / temp_parameter
        for j in range(1, theta.shape[0]):
            c_new = np.dot(theta[[j], :], xt) / temp_parameter
            if c_new > c:
                c = c_new

        # Calulate denominator of every log terms
        sum = 0
        for l in range(0, theta.shape[0]):
            sum = sum + np.exp(np.dot(theta[[l], :], xt) / temp_parameter - c)

        # Calculate all log terms
        for j in range(0, theta.shape[0]):
            if Y[i] == j:
                exp_term = np.exp(np.dot(theta[[j], :], xt) / temp_parameter - c)
                log_term = np.log(exp_term / sum)
                penality = penality + log_term

    n = Y.shape[0]
    penality = -penality / n

    reg = 0
    for j in range(0, theta.shape[0]):
        for i in range(0, X.shape[1]):
            reg = reg + theta[j, i]**2
    reg = reg * lambda_factor / 2

    cost = penality + reg
    return cost

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    cost_grad = np.zeros([theta.shape[0], theta.shape[1]])

    # Iterate through all training data
    for i in range(0, X.shape[0]):
        x = X[[i], :]

        # Calculate the probability distribution of label with given
        # learning vector x
        p_term = compute_probabilities(x, theta, temp_parameter)

        # Iterate through all learning parameters
        for m in range(0, theta.shape[0]):
            if(Y[i] == m):
                p_term[m] = 1 - p_term[m]
            else:
                p_term[m] = -p_term[m]

            # Calculate cost gradient of the m-th label
            cost_grad[[m], :] = cost_grad[[m], :] + x*p_term[m]

    # Scale the cost gradient and add regularization term
    n = Y.shape[0]
    cost_grad = -1 / (temp_parameter * n) * cost_grad
    G = cost_grad + lambda_factor*theta

    # Use sparse matrix to speed up gradient descent
    _G = sparse.coo_matrix(G).toarray()
    new_theta = theta - alpha * _G
    return new_theta

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    return train_y % 3, test_y % 3

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean((assigned_labels % 3) == Y)

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        print("#{}".format(i))
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
