from regression import *
from dataload import *


def main():
    X, y, theta, m, n = load_data("data/ex2data1.txt")

    alpha, num_iters = 0.01, 1500

    theta = gradient_descent(X, y, theta, m, alpha, num_iters)
    print(f"Model accuracy: {accuracy(X, y, theta)}%")


if __name__ == '__main__':
    main()
