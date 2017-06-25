import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

range = np.arange(-10, 10, 0.1)

for mean, standard_deviation in [(-4, 3), (0, 3), (4, 3)]:
    plt.plot(range, norm.pdf(range, mean, standard_deviation))

plt.show()
