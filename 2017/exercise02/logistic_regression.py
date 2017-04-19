import numpy as np
import matplotlib.pyplot as plt
import math


def read_data():
    data = np.genfromtxt('data.txt', delimiter=' ')
    return data


def separate_values(data):
    x = data[:, :2]
    y = data[:, 2]
    return x, y


def classify_data(data):
    c1 = data[:50, :]
    c2 = data[50:, :]
    return c1, c2


def plot_points(c1, c2, theta, theta_first):
    plt.scatter(c1, c2, c=['red', 'blue'])
    x_values = np.arange(-3, 3, 0.000001)
    y_values = (theta[0] + theta[1] * x_values) * ((-1) / theta[2])
    plt.plot(x_values, y_values)
    y_values = (theta_first[0] + theta_first[1] *
                x_values) * ((-1) / theta_first[2])
    plt.plot(x_values, y_values)
    plt.show()


def sigmoid(x):
    den = 1.0 + math.e ** (-1.0 * x)
    d = 1.0 / den
    return d


def gradient_descent(x, y, alpha, theta):
    x_trans = x.transpose()
    for i in range(0, 100000):
        hypothesis = np.dot(x, theta)
        hypothesis = sigmoid(hypothesis)
        loss = hypothesis - y
        gradient = np.dot(x_trans, loss)
        theta = theta - alpha * gradient

    return theta


def show_plot():
    data = read_data()
    x, y = separate_values(data)

    c1, c2 = classify_data(x)
    alpha = 0.001
    theta = np.random.rand(3)
    theta_first = theta
    np_first_column = np.ones(shape=(100, 1))
    x = np.concatenate((np_first_column, x), axis=1)
    theta = gradient_descent(x, y, alpha, theta)
    plot_points(c1, c2, theta, theta_first)

show_plot()