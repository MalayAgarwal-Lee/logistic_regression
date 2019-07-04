from regression import *
from dataload import *


def main():
    X, y, theta, m, n = load_data("data/ex2data1.txt")

    # Learning rate and number of iterations
    alpha, num_iters = 0.01, 1500

    # Running gradient descent
    theta = gradient_descent(X, y, theta, m, alpha, num_iters)
    # Getting model accuracy using accuracy() in regression.py
    print(f"Model accuracy: {accuracy(X, y, theta)}%")


if __name__ == '__main__':
    main()
