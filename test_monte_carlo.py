from random import choice, random
import numpy as np
from math import sin, sqrt, exp, cos, pi
categories = [list(range(200)) for _ in range(2)]


def functi1(x): # Fonseca Fleming
    a = [- 4. + 0.04 * i for i in x]
    suma = 0.0
    for i in range(len(a)):
        suma += (a[i] - 1. / sqrt(2.)) ** 2.
    return 1.0 - exp(- suma)
# functi1 = memoize(functi1)


def functi2(x): # Fonseca Fleming
    a = [- 4. + 0.04 * i for i in x]
    suma = 0.0
    for i in range(len(a)):
        suma += (a[i] + 1. / sqrt(2.)) ** 2.
    return 1.0 - exp(- suma)


def functi3(x): # Kursawe
    a = [- 5. + 0.05 * i for i in x]
    suma = 0.0
    for i in range(len(a) - 1):
        suma += - 10.0 * exp(- 0.2 * sqrt(a[i] ** 2.0 + a[i+1] ** 2.0))
    return suma


def functi4(x): # Kursawe
    a = [- 5. + 0.05 * i for i in x]
    suma = 0.0
    for i in range(len(a)):
        suma += abs(a[i]) ** 0.8 + 5.0 * sin(a[i] ** 3.0)
    return suma


def functi5(x): # Viennet
    a = [- 3. + 0.03 * i for i in x]
    return 0.5 * (a[0] ** 2.0 + a[1] ** 2.0) + sin(a[0] ** 2.0 + a[1] ** 2.0)


def functi6(x): # Viennet
    a = [- 3. + 0.03 * i for i in x]
    return (3.0 * a[0] - 2.0 * a[1] + 4.0) ** 2.0 / 8.0 + (a[0] - a[1] + 1.0) ** 2.0 / 27.0 + 15


def functi7(x): # Viennet
    a = [- 3. + 0.03 * i for i in x]
    return 1.0 / (a[0] ** 2.0 + a[1] ** 2.0 + 1.0) - 1.1 * exp(- (a[0] ** 2.0 + a[1] ** 2.0))


def f1(x): # Bihn Korn
    a = [0.025 * x[0], 0.015 * x[1]]
    return 4.0 * a[0] ** 2.0 + 4.0 * a[1] ** 2.0


def f2(x): # Bihn Korn
    a = [0.025 * x[0], 0.015 * x[1]]
    return (a[0] - 5.0) ** 2.0 + (a[1] - 5.0) ** 2.0


def f1_chakong(x):
    a = [- 20.0 + 0.2 * i for i in x]
    return 2.0 + (a[0] - 2.0) ** 2.0 + (a[1] - 1.0) ** 2.0


def f2_chakong(x):
    a = [- 20.0 + 0.2 * i for i in x]
    return 9.0 * a[0] - (a[1] - 1.0) ** 2.0

#  Poloni
a1 = 0.5 * sin(1.0) - 2.0 * cos(1.0) + sin(2.0) - 1.5 * cos(2.0)
a2 = 1.5 * sin(1.0) - cos(1.0) + 2.0 * sin(2.0) - 0.5 * cos(2.0)
b1 = lambda x, y: 0.5 * sin(x) - 2.0 * cos(x) + sin(y) - 1.5 * cos(y)
b2 = lambda x, y: 1.5 * sin(x) - cos(x) + 2.0 * sin(y) - 0.5 * cos(y)

def f1_poloni(x):
    a = [- pi + pi / 100.0 * i for i in x]
    return 1.0 + (a1 - b1(a[0], a[1])) ** 2.0 + (a2 - b2(a[0], a[1])) ** 2.0


def f2_poloni(x):
    a = [- pi + pi / 100.0 * i for i in x]
    return (a[0] + 3.0) ** 2.0 + (a[1] + 1.0) ** 2.0


# ZDT1
g_zdt = lambda x: 1.0 + (9.0 / 9.0) * sum(x[1:])
h_zdt1 = lambda x: 1.0 - sqrt(x[0] / g_zdt(x))
h_zdt2 = lambda x: 1.0 - (x[0] / g_zdt(x)) ** 2.0
h_zdt3 = lambda x: 1.0 - sqrt(x[0] / g_zdt(x)) - (x[0] / g_zdt(x)) * sin(10.0 * pi * x[0])

def f1_zdt(x):  # ZDT1
    # a = [0.0005 * i for i in x]
    return x[0] * 0.2


def f2_zdt1(x):  # ZDT1
    a = [0.2 * i for i in x]
    return g_zdt(a) * h_zdt1(a)


def f1_osyczka(x):
    a = [0. for i in range(6)]
    a[0] = 0.05 * x[0]
    a[1] = 0.05 * x[1]
    a[5] = 0.05 * x[5]
    a[2] = 1.0 + 0.02 * x[2]
    a[4] = 1.0 + 0.02 * x[4]
    a[3] = 0.03 * x[3]
    return - 25.0 * (a[0] - 2.0) ** 2.0 - (a[1] - 2.0) ** 2.0 - (a[2] - 1.0) ** 2.0 - (a[3] - 4.0) ** 2.0 - (a[4] - 1.0) ** 2.0


def f2_osyczka(x):
    a = [0. for i in range(6)]
    a[0] = 0.05 * x[0]
    a[1] = 0.05 * x[1]
    a[5] = 0.05 * x[5]
    a[2] = 1.0 + 0.02 * x[2]
    a[4] = 1.0 + 0.02 * x[4]
    a[3] = 0.03 * x[3]
    return sum([i ** 2.0 for i in a])


def fonseca1(x):
    a = [- 4.0 + 0.04 * i for i in x]
    suma = 0.0
    for i in range(len(a)):
        suma += (a[i] - 1.0 / sqrt(len(a))) ** 2.0
    return 1.0 - exp(- suma)


def fonseca2(x):
    a = [- 4.0 + 0.04 * i for i in x]
    suma = 0.0
    for i in range(len(a)):
        suma += (a[i] + 1.0 / sqrt(len(a))) ** 2.0
    return 1.0 - exp(- suma)


with open("poloni_all_values.dat", "w") as output:
    for _ in range(100000):
        x = [choice(categories[i]) for i in range(len(categories))]
        # x = [random() for _ in range(30)]
        # x0 = [random()] + [0.0 for _ in range(29)]
        output.write("{} {} {} {}\n".format(f1_poloni(x), f2_poloni(x), - pi + pi / 100.0 * x[0], - pi + pi / 100.0 * x[1]))#, functi7(x)))
        # print h_zdt1(x0)
        # output.write("{} {}\n".format(x0[0], g_zdt1(x0) * h_zdt1(x0)))#, functi7(x)))
