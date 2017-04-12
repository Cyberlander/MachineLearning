import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    y = np.ones(100)
    x = np.random.rand(100)
    x.shape = (100, 1)
    x_first_column = np.ones(100)
    x_first_column.shape = (100, 1)

    for i in range(100):
        offset = np.random.uniform(-0.1, 0.1, size=1)
        y[i] = np.sin(2 * np.pi * x[i]) + offset[0]
    x = np.concatenate((x_first_column, x), axis=1)

    return x, y


def gradient_descent(x, y, alpha, theta):
    x_trans = x.transpose()
    loss_data = []
    loss_sum = 0
    for i in range(0, 100000):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        if i % 1000 == 0:
            loss_square = loss**2
            loss_sum = np.sum(loss_square)
            print( loss_sum )
            loss_data.append( loss_sum )
        gradient = np.dot(x_trans, loss)
        theta = theta - alpha * gradient

    return theta, loss_data


def caluculate_theta(x, y, degree):
    alpha = 0.001

    x_second_column = x[:, 1]

    for i in range(2, degree):
        new_column = x_second_column**i
        new_column.shape = (100, 1)
        x = np.concatenate((x, new_column), axis=1)

    theta = np.ones(degree)
    theta = gradient_descent(x, y, alpha, theta)

    return theta


def show_plot():
    x, y = generate_data()
    degree = 3
    theta,loss_data = caluculate_theta(x, y, degree + 1)
    print( theta )
    print( loss_data )

    plt.plot(x[:, 1], y, "ro")

    x_values = np.arange(0, 1, 0.000001)

    y_values = 0
    for i in range(0,len(theta)):
         y_values += theta[i] * x_values**i

    plt.plot(x_values, y_values)

    plt.axis([0, 1, -1, 1])
    plt.show()
    plot_error( loss_data )


def plot_error( loss_data ):
    x_values = np.arange(0, 100, 1)
    plt.plot( x_values, loss_data )
    plt.axis([0,100, 0, 700 ])
    plt.show()

show_plot()
