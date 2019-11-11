import matplotlib.pyplot as plt
import numpy as np

y = np.arange(-2, 2, 0.1)
x = np.exp(y)

plt.plot(x, y)
plt.show()