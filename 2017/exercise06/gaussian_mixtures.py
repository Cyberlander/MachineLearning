import matplotlib.pyplot as plt
import numpy as np

params = [
    ([0, 0], [[1, 0], [0, 100]], 'go'),
    ([0, 0], [[100, 80], [0, 3]], 'bo'),
    ([0, 0], [[100, -20], [0, 10]], "ro")
]

for mean, cov, point in params:
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    plt.plot(x, y, point)

plt.show()
