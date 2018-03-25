from matplotlib import pyplot as plt


def linear_function(x1, x2, x):
    # print(x1, x2, x, ((x2[1] - x1[1]) / (x2[0] - x1[0])) * (x - x1[0]) + x1[1])
    return ((x2[1] - x1[1]) / (x2[0] - x1[0])) * (x - x1[0]) + x1[1]


def dynamic_weights(T, often):
    weights = [[]]
    for t in range(int(T / 2)):
        if t % (T / 2 / often) == 0:
            i = 0
            num = t
        if i < int((T / 2) / (often * 2)):
            first = linear_function([num, 0.0], [num + (T / 2.) / (often * 2.0), linear_function([0.0, 0.0], [float(T) / 2.0, 1.0], num + T / 2. / (often * 2.0))], t)
            i += 1
            print [0.0, 0.0], [T / 2.0, 1.0], num
            weights[0].append(first)
        else:
            first = linear_function([num + (T / 2.) / (often * 2.0), linear_function([0.0, 0.0], [T / 2.0, 1.0], num + (T / 2.) / (often * 2.0))], [num + (T / 2) / often, 0.0], t)
            weights[0].append(first)
    weights[0] += list(reversed(weights[0]))

    weights.append([])
    for t in range(int(T / 2)):
        if t % (T / 2 / often) == 0:
            i = 0
            num = t
        if i < int((T / 2) / (often * 2)):
            second = linear_function([num, 0.0], [num + int(T / 2) / (often * 2.0), 1.0 - weights[0][num + int((T / 2) / (often * 2))]], t)
            i += 1
            weights[1].append(second)
        else:
            second = linear_function([num + int(T / 2) / (often * 2.0), 1.0 - weights[0][num + int((T / 2) / (often * 2))]], [num + int(T / 2) / often, 0.0], t)
            weights[1].append(second)
    weights[1] += list(reversed(weights[1]))

    weights.append([])
    for t in range(int(T / 2)):
        third = 1.0 - weights[0][t] - weights[1][t]
        weights[2].append(third)
    weights[2] += list(reversed(weights[2]))

    return [weights[0] + weights[1] + weights[2], weights[1] + weights[2] + weights[0], weights[2] + weights[0] + weights[1]]


if __name__ == '__main__':
    T = 120
    n_w = 2  # Needs to be such that T / 2 / n_w / 2 is an integer. T / (n_w * 4).
    ans = dynamic_weights(T, n_w)
    with open("dyn_weights_good.dat", "w") as outf:
        for i in range(len(ans[0])):
            outf.write("{} {} {}\n".format(ans[0][i], ans[1][i], ans[2][i]))
    print(sum(ans[0]), sum(ans[1]), sum(ans[2]))

    # plt.figure()
    # plt.plot(list(range(len(ans[0]))), ans[0], c='red')
    # plt.plot(list(range(len(ans[1]))), ans[1], c='orange')
    # plt.plot(list(range(len(ans[2]))), ans[2])
    # plt.show()
