import numpy as np
from matplotlib import pyplot as plt, style
# from mpl_toolkits.mplot3d import Axes3D
from random import randint, random
from copy import deepcopy
from statistics import mode, mean, StatisticsError
from functools import reduce
from joblib import Parallel, delayed
from pareto_epsilon import eps_sort
from math import sqrt, sin

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


def function1(x):
    return 1.0 / len(x) * sum([item ** 2.0 for item in x])
function1 = memoize(function1)


def function2(x):
    return 1.0 / len(x) * sum([(item - 2.0) ** 2.0 for item in x])


def function3(x):
    return 1.0 / len(x) * sum([item ** 2.0 for item in x])


def function4(x):
    return 1.0 / len(x) * sum([sin(item) for item in x])


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


def dominates(candidate, particle):
    return sum([candidate[x] <= particle[x] for x in range(len(candidate))]) == len(candidate)


def dominate_one_in_swarm(candidate, swarm):
    for particle in swarm:
        if dominates(candidate, particle):
            flag = True
            break
    else:
        flag = False
    return flag


def dominates_any(candidate, archive):
    for particle in archive:
        if dominates(candidate, particle):
            out = particle
            break
    else:
        out = None
    return out


def dominated(archive):
    for particle1 in archive:
        for particle2 in archive:
            if dominates(particle1, particle2):
                out = particle2
                break
        break
    else:
        out = None
    return out


def not_dominated(candidate, archive):
    for particle in archive:
        if dominates(particle, candidate):
            flag = True
            break
    else:
        flag = False
    return flag


def distance_multidimension(x, y):
    sum_ = 0
    for i in range(len(x)):
        sum_ += (x[i] - y[i]) ** 2.0
    return sqrt(sum_)


def similarity(candidate, archive, tolerance):
    for particle in archive:
        if distance_multidimension(candidate, particle) < tolerance:
            flag = True
            break
    else:
        flag = False
    return flag


def replace(list_, old, new):
    for i, v in enumerate(list_):
        if v == old:
            list_.pop(i)
            list_.insert(i, new)


def eliminate(list_, to_be_deleted):
    for i, v in enumerate(list_):
        if v == to_be_deleted:
            list_.pop(i)


class PSOCategorical:
    def __init__(self, n_particles, n_discrete_vars, scaling_factor):
        self.weight_local = 0.729
        self.weight_global = 1.49618
        self.inertia_weight = 1.49618
        self.n_iterations = 1000
        self.n_samples = 5
        self.n_particles = n_particles
        self.scaling_factor = scaling_factor
        self.n_discrete_vars = n_discrete_vars
        self.bins = randint(2, 25)
        self.real_angle = np.random.choice([30.0, 60.0, 90.0, 120.0, 180.0])
        self.artif_angle = 190.0
        while self.artif_angle > self.real_angle:
            self.artif_angle = np.random.choice([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0])
        self.categories = [list(range(3)), list(range(3)), list(range(6)), list(range(4)), list(range(4)),
                           list(range(4)), list(range(5)), list(range(4)), list(range(2)), list(range(2))]
        self.positions_categorical = [[[0 for _ in var] for var in self.categories] for _ in range(self.n_particles)]
        self.velocities_categorical = [[[0 for _ in var] for var in self.categories] for _ in range(self.n_particles)]
        self.positions_discrete = [[0 for _ in range(self.n_discrete_vars)] for _ in range(self.n_particles)]
        self.velocities_discrete = [[0 for _ in range(self.n_discrete_vars)] for _ in range(self.n_particles)]
        self.local_best_fitness = [999999.9 for _ in range(self.n_particles)]
        self.global_best_fitness = 999999.9
        self.local_best = []
        self.global_best = []
        self.samples = [[] for _ in range(self.n_particles)]
        self.fitness = [[999999.9, 999999.9] for _ in range(self.n_particles)]
        self.obj_function = [0.0 for _ in range(self.n_particles)]
        self.archive = []
        self.archive_size = 1000
        self.old_swarm = []

    def update_archive(self, swarm):
        for particle in swarm:
            if dominate_one_in_swarm(particle, self.old_swarm) and not_dominated(particle, self.archive) is False and similarity(particle, self.archive, 1e-5) is False:
                to_be_changed = dominates_any(particle, self.archive)
                to_be_substituted = dominated(self.archive)
                if len(self.archive) < self.archive_size:
                    self.archive.append(particle)
                elif to_be_changed:
                    replace(self.archive, to_be_changed, particle)
                elif to_be_substituted:
                    replace(self.archive, to_be_substituted, particle)

        for solution1 in self.archive:
            for solution2 in self.archive:
                if solution1 != solution2:
                    if dominates(solution1, solution2):
                        eliminate(self.archive, solution2)

    def representative_sample(self, distribution):
        samples = []
        for _ in range(self.n_samples):
            samples.append(self.sample_distribution(distribution))
        samples = np.array(samples)
        representative_sample = [mode_custom(samples[:, i]) for i in range(len(self.categories))]
        return representative_sample

    def fitness_function(self, distribution, fitness_functions):
        samples = []
        for _ in range(self.n_samples):
            samples.append(self.sample_distribution(distribution))
        means = [mean([fitness_function(sample) for sample in samples]) for fitness_function in fitness_functions]
        return means

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
                    self.velocities_categorical[particle][var][prob] = self.inertia_weight * \
                                                                       self.velocities_categorical[particle][var][prob] \
                                                                       + self.weight_global * random() * \
                                                                       (self.global_best[var][prob] -
                                                                       self.positions_categorical[particle][var][prob])\
                                                                       + self.weight_local * random() * \
                                                                       (self.local_best[particle][var][prob] -
                                                                        self.positions_categorical[particle][var][prob])

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        from time import time
        from math import sin, pi
        start = time()
        best_global = []
        self.initialise_categorical_positions()
        self.initialise_categorical_velocities()
        self.calculate_new_velocities()
        self.update_position()
        self.n_iterations = 1000
        self.n_samples = 1
        self.archive = deepcopy(self.fitness)
        for iteration in range(self.n_iterations):
            weight1 = sin(4.0 * pi * iteration / self.n_iterations)
            if weight1 == 0.0:
                weight1 = 0.00001
            elif weight1 == 1.0:
                weight1 = 0.99999
            weight2 = 1.0 - weight1
            self.samples = [self.representative_sample(position) for position in self.positions_categorical]
            self.old_swarm = deepcopy(self.fitness)
            self.fitness = [self.fitness_function(position, [function3, function4]) for position in self.positions_categorical]
            # self.fitness = Parallel(n_jobs=-1)(delayed(self.fitness_function)(position) for position in self.positions_categorical)
            self.function_values = []
            for particle in range(self.n_particles):
                self.obj_function[particle] = weight1 * self.fitness[particle][0] + weight2 * self.fitness[particle][1]

            self.update_archive(self.fitness)

            for particle in range(self.n_particles):
                if self.obj_function[particle] < self.local_best_fitness[particle]:
                    self.update_local_best(particle, self.positions_categorical[particle], self.samples[particle])
                    self.local_best_fitness[particle] = self.obj_function[particle]

            max_local = max(self.local_best_fitness)
            if max_local < self.global_best_fitness:
                self.global_best_fitness = max_local
                self.update_global_best(self.local_best[self.local_best_fitness.index(max_local)],
                                        self.representative_sample(
                                            self.local_best[self.local_best_fitness.index(max_local)]))
            best_global.append(self.global_best_fitness)

            plt.cla()
            # ax.set_ylim([0, 5])
            # ax.set_xlim([0, 10])
            ax.scatter(*zip(*self.archive))
            plt.pause(0.01)
        with open("optimiser_output2.dat", "w") as out:
            for item in self.archive:
                out.write("{} {}\n".format(item[0], item[1]))
        print(time() - start, "seconds")
        while True:
            plt.pause(0.05)


if __name__ == '__main__':
    opt = PSOCategorical(10, 0, 0.9)
    opt.run()
