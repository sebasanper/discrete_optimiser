import numpy as np
from matplotlib import pyplot as plt, style
from random import randint, random
from copy import deepcopy
from statistics import mode, mean, StatisticsError
from functools import reduce
# from joblib import Parallel, delayed

style.use('ggplot')


def memoize(fn):
    """returns a memoized version of any function that can be called
    with the same list of arguments.
    Usage: foo = memoize(foo)"""

    def handle_item(x):
        if isinstance(x, dict):
            return make_tuple(sorted(x.items()))
        elif hasattr(x, '__iter__'):
            return make_tuple(x)
        else:
            return x

    def make_tuple(L):
        return tuple(handle_item(x) for x in L)

    def foo(*args, **kwargs):
        items_cache = make_tuple(sorted(kwargs.items()))
        args_cache = make_tuple(args)
        if (args_cache, items_cache) not in foo.past_calls:
            foo.past_calls[(args_cache, items_cache)] = fn(*args, **kwargs)
        return foo.past_calls[(args_cache, items_cache)]

    foo.past_calls = {}
    foo.__name__ = 'memoized_' + fn.__name__
    return foo


def mode_custom(x):
    i = 0
    while True:
        try:
            x_mode = mode(x)
            break
        except StatisticsError:
            x = np.append(x, i)
            i += 1
    return x_mode


def generate_weights(num_weights):
    random_nums = np.random.random(num_weights - 1)
    random_nums[::-1].sort()
    weights = [1.0 - random_nums[0]]
    for i in range(1, num_weights - 1):
        weights.append(random_nums[i - 1] - random_nums[i])
    weights.append(random_nums[-1])
    return weights


def normalise(vector):
    summation = sum(vector)
    vector = [n / summation for n in vector]
    return vector


class PSOCategorical:
    def __init__(self, fit_function, n_particles, n_discrete_vars, scaling_factor):
        self.weight_local = 1.49618
        self.weight_global = 1.49618
        self.inertia_weight = 0.729
        self.n_iterations = 500
        self.n_samples = 10
        self.n_particles = n_particles
        self.scaling_factor = scaling_factor
        self.n_discrete_vars = n_discrete_vars
        self.fit_function = fit_function
        # self.bins = randint(2, 25)
        # self.real_angle = np.random.choice([30.0, 60.0, 90.0, 120.0, 180.0])
        # self.artif_angle = 190.0
        # while self.artif_angle > self.real_angle:
        #     self.artif_angle = np.random.choice([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0])
        # self.categories = [list(range(3)), list(range(3)), list(range(6)), list(range(4)), list(range(4)),
        #                    list(range(4)), list(range(5)), list(range(4)), list(range(2)), list(range(2))]
        self.categories = [list(range(1000)), list(range(1000))]
        self.positions_categorical = [[[0 for _ in var] for var in self.categories] for _ in range(self.n_particles)]
        self.velocities_categorical = [[[0 for _ in var] for var in self.categories] for _ in range(self.n_particles)]
        self.positions_discrete = [[0 for _ in range(self.n_discrete_vars)] for _ in range(self.n_particles)]
        self.velocities_discrete = [[0 for _ in range(self.n_discrete_vars)] for _ in range(self.n_particles)]
        self.local_best_fitness = [999999.0 for _ in range(self.n_particles)]
        self.global_best_fitness = 999999.0
        self.local_best = []
        self.global_best = []
        self.samples = []
        self.fitness = []

    def representative_sample(self, distribution):
        samples = []
        for _ in range(self.n_samples):
            samples.append(self.sample_distribution(distribution))
        samples = np.array(samples)
        representative_sample = [mode_custom(samples[:, i]) for i in range(len(self.categories))]
        return representative_sample

    def fitness_function(self, sample):
        return self.fit_function(sample)
        # samples = []
        # for _ in range(self.n_samples):
        #     samples.append(self.sample_distribution(distribution))
        # return mean([self.fit_function(sample) for sample in samples])

    def initialise_categorical_positions(self):
        for particle in range(self.n_particles):
            for variable in range(len(self.categories)):
                weights = generate_weights(len(self.categories[variable]))
                self.positions_categorical[particle][variable] = weights
        self.local_best = deepcopy(self.positions_categorical)
        self.global_best = deepcopy(self.local_best[0])

    def initialise_categorical_velocities(self):
        for particle in range(self.n_particles):
            for variable in range(len(self.categories)):
                velocities = [0.0 for _ in self.categories[variable]]
                self.velocities_categorical[particle][variable] = velocities

    def update_position(self):
        for particle in range(self.n_particles):
            for var in range(len(self.categories)):
                for prob in range(len(self.categories[var])):
                    self.positions_categorical[particle][var][prob] += self.velocities_categorical[particle][var][prob]
                    if self.positions_categorical[particle][var][prob] > 1:
                        self.positions_categorical[particle][var][prob] = 1.0
                    elif self.positions_categorical[particle][var][prob] < 0:
                        self.positions_categorical[particle][var][prob] = 0.0

        for part in range(self.n_particles):
            for var in range(len(self.categories)):
                self.positions_categorical[part][var] = normalise(self.positions_categorical[part][var])

    def update_global_best(self, position, sample):
        for var in range(len(self.categories)):
            for prob in range(len(self.categories[var])):
                if prob != sample[var]:
                    self.global_best[var][prob] = self.scaling_factor * position[var][prob]
                else:
                    summation = sum([(1.0 - self.scaling_factor) * pr if position[var].index(pr) != prob else 0 for pr
                                     in position[var]])
                    self.global_best[var][prob] = position[var][prob] + summation

    def update_local_best(self, particle, position, sample):
        for var in range(len(self.categories)):
            for prob in range(len(self.categories[var])):
                if prob != sample[var]:
                    self.local_best[particle][var][prob] = self.scaling_factor * position[var][prob]
                else:
                    summation = sum([(1.0 - self.scaling_factor) * pr if position[var].index(pr) != prob else 0 for pr
                                     in position[var]])
                    self.local_best[particle][var][prob] = position[var][prob] + summation

    def sample_distribution(self, distribution):
        sample = [np.random.choice(self.categories[var], p=distribution[var]) for var in range(len(distribution))]
        return sample

    def calculate_new_velocities(self):
        for particle in range(self.n_particles):
            for var in range(len(self.categories)):
                for prob in range(len(self.categories[var])):
                    if random() < 0.2:
                        turbulence = random()
                    else:
                        turbulence = 0.0
                    self.velocities_categorical[particle][var][prob] = self.inertia_weight * \
                                                                       self.velocities_categorical[particle][var][prob] \
                                                                       + self.weight_global * random() * \
                                                                         (self.global_best[var][prob] -
                                                                          self.positions_categorical[particle][var][
                                                                              prob]) + \
                                                                       self.weight_local * random() * \
                                                                       (self.local_best[particle][var][prob] -
                                                                        self.positions_categorical[particle][var][prob]) + turbulence

    def run(self):
        from time import time
        start = time()
        best_global = []
        self.initialise_categorical_positions()
        self.initialise_categorical_velocities()
        fileout = open("convergence.dat", "w")
        for iteration in range(self.n_iterations):
            self.calculate_new_velocities()
            self.update_position()
            # print ("\n --- Iteration ")
            self.samples = [self.representative_sample(position) for position in self.positions_categorical]
            self.fitness = [self.fitness_function(position) for position in self.samples]
            # print("Samples --------")
            # print (self.samples)
            # print("\n Fitness -----------")
            # print (self.fitness)
            # self.fitness = Parallel(n_jobs=2)(delayed(self.fitness_function)(position) for position in self.positions_categorical)

            for particle in range(self.n_particles):
                if self.fitness[particle] < self.local_best_fitness[particle]:
                    self.update_local_best(particle, self.positions_categorical[particle], self.samples[particle])
                    self.local_best_fitness[particle] = self.fitness[particle]
            # print("\n ---- Local best fitness, max -----------")
            max_local = min(self.local_best_fitness)
            # print(self.local_best_fitness, max_local)
            # print("\n ---- Local best particles, index -----------")
            max_local_idx = self.local_best_fitness.index(max_local)
            # print(self.local_best_fitness.index(max_local), self.samples[max_local_idx])
            if max_local < self.global_best_fitness:
                self.global_best_fitness = max_local
                xx = self.samples[max_local_idx]
                self.update_global_best(self.local_best[self.local_best_fitness.index(max_local)],
                                        xx)
                fileout.write("{} {} {} {}\n".format(iteration, self.global_best_fitness, - 5.0 + 0.1 * xx[0], - 5.0 + 0.1 * xx[1]))

            # print("\n ---- Global best fitness, particle -----------")
            print(iteration, self.global_best_fitness, xx)
            best_global.append(self.global_best_fitness)
        fileout.close()
        print(time() - start, "seconds")
        plt.figure()
        plt.plot(best_global)
        plt.show()
        return self.global_best_fitness, self.global_best, xx


if __name__ == '__main__':
    def rosenbrock(x):
        x1 = [- 5.0 + 0.01 * x[0], - 5.0 + 0.01 * x[1]]
        fun = (1.0 - x1[0]) ** 2.0 + 100.0 * (x1[1] - x1[0] ** 2.0) ** 2.0
        return fun
    rosenbrock = memoize(rosenbrock)
    # with open("rosenbrock.dat", "w") as rosout:
    #     for x in range(100):
    #         for y in range(100):
    #             x1, ans = rosenbrock([x, y])
    #             rosout.write("{} {} {} {} {}\n".format(x, y, x1[0], x1[1], ans))

    opt = PSOCategorical(rosenbrock, 25, 0, 0.75)
    fit, vec, best_vec = opt.run()
    plt.figure()
    my_xticks = [item for sublist in opt.categories for item in sublist]
    plt.xticks(list(range(2000)), my_xticks)
    plt.plot([item for sublist in vec for item in sublist])
    plt.show()
