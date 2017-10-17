from random import choice
import numpy as np
from math import sin
categories = [list(range(3)), list(range(3)), list(range(6)), list(range(4)), list(range(4)),
                           list(range(4)), list(range(5)), list(range(4)), list(range(2)), list(range(2))]


def function1(x):
    return 1.0 / len(x) * sum([float(item) ** 2.0 for item in x])


def function2(x):
    return 1.0 / len(x) * sum([(float(item) - 2.0) ** 2.0 for item in x])


def function3(x):
    return 1.0 / sum(x)**2.0


def function4(x):
    return 1.0 / len(x) * sum([sin(item) for item in x])

with open("all_values.dat", "w") as output:
    for _ in range(10000):
        x = [choice(categories[i]) for i in range(len(categories))]
        output.write("{} {} {}\n".format(function1(x), function2(x), function3(x)))
