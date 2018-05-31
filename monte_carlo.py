from random import choice, random, randint
import numpy as np
from math import sin
categories = [list(range(3)), list(range(3)), list(range(6)), list(range(4)), list(range(4)),
                           list(range(4)), list(range(5)), list(range(4)), list(range(2)), list(range(2))]


def function1(x):
    return (x[0] - 2.0) ** 2.0 / 2.0 + (x[1] + 1.0) ** 2.0 / 13.0 + 3.0


def function2(x):
    return (x[0] + x[1] - 3.0) ** 2.0 / 175.0 + (2.0 * x[1] - x[0]) ** 2.0 / 17.0 - 13.0


def function3(x):
    return (3.0 * x[0] - 2.0 * x[1] + 4.0) ** 2.0 / 8.0 + (x[0] - x[1] + 1.0) ** 2.0 / 27.0 + 15.0


def function4(x):
    return (3.0 * x[0] + x[1] + 9.0) ** 2.0 / 34.0 + (x[0] + 1.0) ** 2.0 / 15.0 + 29.0


with open("all_values_4D_noconst_int.dat", "w") as output:
    for _ in range(1400):
        x = [- 4.0 + randint(0,200) * 0.04, - 4.0 + randint(0,200) * 0.04]
        # while 4.0 * x[0] + x[1] - 4 > 0.0 or -1.0 - x[0] > 0.0 or x[0] - x[1] - 2.0 > 0.0:
        #     x = [- 4.0 + random() * 8.0, - 4.0 + random() * 8.0]
        # x = [choice(categories[i]) for i in range(len(categories))]
        output.write("{} {} {} {} {} {}\n".format(function1(x), function2(x), function3(x), function4(x), x[0], x[1]))
