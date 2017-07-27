import numpy as np
from statistics import mode

a = np.array([[2, 3], [4, 2], [5, 5], [6, 3], [2, 4]])
print(mode(a[:, 1]))
