import numpy as np
from statistics import mode, StatisticsError


def mode_custom(x):
    i = 0
    while True:
        try:
            x_mode = mode(x)
            break
        except StatisticsError:
            np.append(x, i)
            print(x)
            i += 1
    return x_mode

a = np.array([([2, 3, 1]), ([4, 5])])
print(a.tolist())

# b = [4, 5, 6]
# print(b)
# b = np.array(b)
# b = np.append(b, [3, 3, 3])
# print(b)
