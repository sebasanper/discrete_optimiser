from numpy import sqrt

dist = []
nadir = []
name=[]
dist_norm = []

def normalise(x, min_x, max_x):
	return (x - min_x) / (max_x - min_x)


with open("../WINDOW_dev/pareto_optimisers.dat", "r") as infile:
	with open("../WINDOW_dev/pareto_optimisers_normal.dat", "w") as out:
		for line in infile:
			cols = line.split()
			name.append(cols[0])
			dist_norm.append(sqrt(normalise(float(cols[1]), 7.5007, 11.6251) ** 2.0 + normalise(float(cols[2]), 0.0, 1.9595) ** 2.0 + normalise(float(cols[3]), 0.0029, 0.2) ** 2.0))
			out.write("{} {} {} {}\n".format(cols[0], normalise(float(cols[1]), 7.5007, 11.6251), normalise(float(cols[2]), 0.0, 1.9595), normalise(float(cols[3]), 0.0029, 0.2)))
			dist.append(sqrt((float(cols[1]) - 7.5007) ** 2.0 + (float(cols[2]) - 0.0) ** 2.0 + (float(cols[3]) - 0.0029) ** 2.0))
			nadir.append(sqrt((float(cols[1]) - 11.6251) ** 2.0 + (float(cols[2]) - 1.9595) ** 2.0 + (float(cols[3]) - 0.2) ** 2.0))

b = zip(nadir, name)
a = zip(dist, name)
c = zip(dist_norm, name)
a.sort()
c.sort()
b.sort(reverse=True)
print a
print
print b
print
print c
