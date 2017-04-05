import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    # y = np.zeros(100)
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

    for i in range(0, 100000):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(x_trans, loss)
        theta = theta - alpha * gradient

    return theta


theta = np.ones(4)
alpha = 0.001

x, y = generate_data()
x_second_column = x[:,1]

for i in range(2,4):
    new_column = x_second_column**i
    new_column.shape = (100, 1)
    x = np.concatenate((x, new_column), axis=1)


# x_third_column = x_second_column**2
# x_third_column.shape = (100,1)
# x = np.concatenate((x, x_third_column), axis=1)

theta = gradient_descent(x, y, alpha, theta)

plt.plot(x[:,1], y, "ro")

x_function = np.arange(0,1,0.000001)
plt.plot(x_function, theta[0] + theta[1] * x_function + theta[2] * x_function**2 + theta[3] * x_function**3)


plt.axis([0, 1, -1, 1])
plt.show()