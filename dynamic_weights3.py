from matplotlib import pyplot as plt
from numpy import pi, sin, cos

weights = [[None for _ in range(200)] for _ in range(3)]

def fun(x):
    return abs(sin(2.0 * pi * x / 200.0)), abs(cos(2.0 * pi * x / 200.0))

F = 50
w1 = []
w2 = []
w3 = []
for t1 in range(16*F):
	w11 = abs(cos(2.0*pi*t1/F))
	w22 = 0.5+ 0.5*(sin(2.0*pi*t1/F))
	w33 = abs(sin(2.0*pi*t1/F))
	w1.append(w11/(w11+w22+w33))
	w2.append(w22/(w11+w22+w33))
	w3.append(w33/(w11+w22+w33))
	# w1.append(w11)
	# w2.append(w22)
	# w3.append(w33)

print(sum(w1), sum(w2), sum(w3))
plt.figure()
plt.plot(list(range(len(w1))), w1, c='red')
plt.plot(list(range(len(w2))), w2, c='green')
plt.plot(list(range(len(w3))), w3)
plt.show()
