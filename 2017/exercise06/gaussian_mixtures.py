import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

params = [
    ([0, 0], [[1, 0], [0, 100]]),
    ([0, 0], [[100, 80], [0, 3]]),
    ([0, 0], [[100, -20], [0, 10]])
]

guess = {
    'mean': [1,1]
    'cov': [[1,0],[0,1]],
    'class': [0.3, 0.7]
}


def generate_data(params):
    data = []

    for mean, covin params:
        x, y = np.random.multivariate_normal(mean, cov, 100).T
        data.append([x,y])

    return data


def show_plot(data):
    colors = ['ro','bo','go']

    for i in range(len(data)):
        x, y = data[i][0], data[i][1]
        plt.plot(x, y, colors[i])

    plt.show()


def probability(values, mean, cov, p):
    prob = p
    for i in range(len(values)):
        prob *= norm.pdf(values[i], mean[i], cov[i][i])

    return prob


data = generate_data(params)
show_plot(data)
