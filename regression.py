import numpy as np
import matplotlib.pyplot as plt


def sigmoid(Z):
    exponent = 1 + np.exp(np.negative(Z))
    return np.reciprocal(exponent)


def cost_function(X, y, theta, m):
    predictions = sigmoid(X @ theta)
    left_opr = y * np.log(predictions)
    right_opr = (1 - y) * np.log(1 - predictions)
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
