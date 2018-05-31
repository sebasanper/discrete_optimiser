from matplotlib import pyplot as plt
from numpy import pi, sin, cos

# weights = [[None for _ in range(200)] for _ in range(3)]

# def fun(x):
#     return abs(sin(2.0 * pi * x / 200.0)), abs(cos(2.0 * pi * x / 200.0))

F = 100
w1 = []
w2 = []
w3 = []
w4 = []
w5 = []
w6 = []
for t1 in range(5 * F):
	w11 = 0.5+0.5*(sin(2.0*pi*t1/F))
	w22 = (1.0 - w11) * (0.5+0.5*(sin(2.0*pi*t1/F)))
	w33 = 1.0 - w22 - w11 #0.5+0.5*(sin(2.0*pi*t1/(F) + 4.0*pi/5.0))
	# w44 = 0.5+0.5*(sin(2.0*pi*t1/(F) + 6.0*pi/5.0))
	# w55 = 0.5+0.5*(sin(2.0*pi*t1/(F) + 8.0*pi/5.0))
	# w1.append(w11/(w11+w22+w33+w44+w55+w66))
	# w2.append(w22/(w11+w22+w33+w44+w55+w66))
	# w3.append(w33/(w11+w22+w33+w44+w55+w66))
	# w4.append(w44/(w11+w22+w33+w44+w55+w66))
	# w5.append(w55/(w11+w22+w33+w44+w55+w66))
	w1.append(w11)
	w2.append(w22)
	w3.append(w33)
	# w4.append(w44)
	# w5.append(w55)

print(sum(w1), sum(w2), sum(w3))#, sum(w4), sum(w5))
plt.figure()
plt.plot(list(range(len(w1))), w1)
plt.plot(list(range(len(w2))), w2)
plt.plot(list(range(len(w3))), w3)
# plt.plot(list(range(len(w4))), w4)
# plt.plot(list(range(len(w5))), w5)
plt.show()
