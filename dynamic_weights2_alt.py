from matplotlib import pyplot as plt
from numpy import pi, sin, cos

weights = [[None for _ in range(200)] for _ in range(3)]

def fun(x):
    return abs(sin(2.0 * pi * x / 200.0)), abs(cos(2.0 * pi * x / 200.0))

F = 50
w1 = []
w2 = []
w3 = []
t1=0
t2=0
t=t1+t2
for t1 in range(F/2):
	for t2 in range(F/2):
		w11 = abs(sin(2*pi*t1/F))
		w1.append(w11)
		w22 = (1.0-w11) * abs(sin(2*pi*t2/F))
		w2.append(w22)
		w3.append(1.0-w11-w22)

print(sum(w1), sum(w2), sum(w3))
plt.figure()
plt.plot(list(range(len(w1))), w1, c='red')
plt.plot(list(range(len(w2))), w2, c='green')
plt.plot(list(range(len(w3))), w3)
plt.show()
