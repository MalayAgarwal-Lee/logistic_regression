import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    '''
    Computes the sigmoid function given as:
        sigmoid(Z) = 1/(1 + e^(-Z))

    In a simple hypothesis such as that for linear regression
    The values can be greater than 1
    But, in logistic regression, the values need to be 0 or 1
    The sigmoid function always yields values between 0 and 1
    Therefore, the function is applied on the simple hypothesis
    This ensures that the hypothesis' range is between 0 and 1

    Arguments:
        Z: scalar or array-like
           Value(s) for which sigmoid needs to be computed

    Returns:
        sigmoid(Z): scalar or array-like
                    Value(s) of sigmoid function for Z
    '''
    exponent = 1 + np.exp(np.negative(Z))
    return np.reciprocal(exponent)


def cost_function(X, y, theta, m):
    '''
    Calculates the value of the cost function
    For a given X, y, theta and m

    If sigmoid(X * theta) are the predicted values for given theta,
    Then the formula for the cost function is as follows:
        J = (-1 / m) * sum(y * log(predictions) + (1 - y) * log(1 - predictions))
        Where the multiplication is done element-wise

    Arguments:
        X: ndarray, (m, n) matrix consisting of the features
        y: ndarray, (m, 1) matrix consisting of y-values
        theta: ndarray, (n, 1) matrix with parameter values
        m: int, number of training examples in dataset

    Returns:
        J: ndarray, (1, 1) matrix containing the cost
    '''

    # Getting predicted values
    predictions = sigmoid(X @ theta)

    # (y * log(predictions))
    # The '*' operator does element-wise multiplication
    left_opr = y * np.log(predictions)

    # ((1 - y) * log(1 - predictions))
    right_opr = (1 - y) * np.log(1 - predictions)

    # Returning the computed cost
    return (-1 / m) * sum(left_opr + right_opr)


def plot_cost(costs):
    '''
    Plots the values of the cost function
    Against number of iterations
    If gradient descent has converged, graph flattens out
    And becomes constant near the final iterations
    Otherwise, it shows a different trend
    '''
    plt.plot(costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("J(theta)")
    plt.title("Iterations vs Cost")
    plt.show()


def gradient_descent(X, y, theta, m, alpha, num_iters):
    j_vals = np.zeros((num_iters, 1))

    for i in range(num_iters):
        difference = np.transpose(sigmoid(X @ theta) - y)
        delta = np.transpose(difference @ X)
        theta = theta - (alpha / m) * delta
        j_vals[i][0] = cost_function(X, y, theta, m)

    plot_cost(j_vals)
    return theta


def predict(X, theta):
    predictions = np.zeros((X.shape[0], 1))
    probs = sigmoid(X @ theta)
    predictions[probs >= 0.5] = 1
    return predictions


def accuracy(X, y, theta):
    predictions = predict(X, theta)
    correct = predictions == y
    return (np.sum(correct) / len(X)) * 100
