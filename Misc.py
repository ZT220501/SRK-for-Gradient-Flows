import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
from numpy import float64
import time

import utils



# test = [[1, 2, 3], [4, 5, 6]]
# test = np.array(test)
# print(test)
# print(test[:, 0])
# print(sum(test[:, 0]))
# print(sum(test[:, 1]))
# print(sum(test))


# arr = np.array([-5, -4, -3, 1, 2])
# print(np.maximum(arr, 0))


# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(utils.half_grid(arr))


# u_test = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
# print(u_test[::2, ::2])



# Grid spacing
fine_dx = 0.02
coarse_dx = 0.1

# 1D non-uniform grid for x and y
x_left   = np.arange(-1.0, -0.5, fine_dx)
x_center = np.arange(-0.5,  0.5 + coarse_dx, coarse_dx)
x_right  = np.arange(0.5 + fine_dx, 1.0 + fine_dx, fine_dx)
x = np.concatenate((x_left, x_center, x_right))

y_left   = np.arange(-1.0, -0.5, fine_dx)
y_center = np.arange(-0.5,  0.5 + coarse_dx, coarse_dx)
y_right  = np.arange(0.5 + fine_dx, 1.0 + fine_dx, fine_dx)
y = np.concatenate((y_left, y_center, y_right))

# 2D meshgrid
X, Y = np.meshgrid(x, y, indexing='xy')

# Plot to visualize
plt.figure(figsize=(6,6))
plt.plot(X, Y, 'k.', markersize=1)
plt.title("Non-uniform mesh with coarse center [-0.5, 0.5]^2")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()

