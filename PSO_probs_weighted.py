import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint, random
from copy import deepcopy
from statistics import mode, mean, StatisticsError


def function1(x):
    return sum(x)


def function2(x):
    return 1.0 / sum(x)


def function3(x):
    return abs(np.average(x) - 2.5)


def function4(x):
    return - sum(x)


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


def generateWeights(num_weights):
    random_nums = np.random.random(num_weights - 1)
    random_nums[::-1].sort()
    weights = [1.0 - random_nums[0]]
    for i in range(1, num_weights - 1):
        weights.append(random_nums[i-1] - random_nums[i])
    weights.append(random_nums[-1])
    # weights = np.array(weights)
    return weights


class PSOCategorical:

    def __init__(self, fit_function, n_particles, n_discrete_vars, scaling_factor):
        self.weight_local = 0.729
        self.weight_global = 1.49618
        self.inertia_weight = 1.49618
        self.n_iterations = 5
        self.n_samples = 30
        self.n_particles = n_particles
        self.scaling_factor = scaling_factor
        self.n_discrete_vars = n_discrete_vars
        self.fit_function = fit_function
        self.bins = randint(2, 25)
        self.real_angle = np.random.choice([30.0, 60.0, 90.0, 120.0, 180.0])
        self.artif_angle = 190.0
        while self.artif_angle > self.real_angle:
            self.artif_angle = np.random.choice([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0])
        self.categories = [list(range(3)), list(range(3)), list(range(6)), list(range(4)), list(range(4)), list(range(4)), list(range(5)), list(range(4)), list(range(2)), list(range(2))]
        self.positions = np.array([[[0 for _ in var] for var in self.categories] for _ in range(self.n_particles)])
        self.velocities = np.array([[[0 for _ in var] for var in self.categories] for _ in range(self.n_particles)])
        self.local_best = deepcopy(self.positions)
        self.local_best_fitness = [999999.0 for _ in range(self.n_particles)]
        self.global_best = deepcopy(self.local_best[0])
        self.global_best_fitness = 999999.0

    def representative_sample(self, distribution):
        samples = []
        for _ in range(self.n_samples):
            samples.append(self.sample_distribution(distribution))
        samples = np.array(samples)
        representative_sample = [mode_custom(samples[:, i]) for i in range(len(self.categories))]
        return representative_sample

    def fitness_function(self, distribution):
        samples = []
        for _ in range(self.n_samples):
            samples.append(self.sample_distribution(distribution))
        return mean([self.fit_function(sample) for sample in samples])

    def normalise(self, vector):
        summation = sum(vector)
        vector = np.array([n / summation for n in vector])
        return vector

    def initialise_categorical_positions(self):
        for particle in range(self.n_particles):
            for variable in range(len(self.categories)):
                weights = generateWeights(len(self.categories[variable]))
                self.positions[particle][variable] = weights

    def initialise_categorical_velocities(self):
        for particle in range(self.n_particles):
            for variable in range(len(self.categories)):
                velocities = np.array([random() for _ in self.categories[variable]])
                self.velocities[particle][variable] = velocities

    def update_position(self):
        self.positions += self.velocities
        self.positions = np.array([[[1.0 if prob > 1.0 else 0.0 if prob < 0.0 else prob for prob in var]
                            for var in part]
                           for part in self.positions])
        self.positions = np.array([[self.normalise(var) for var in part] for part in self.positions])

    def update_global_best(self, position, sample):
        for var in range(len(self.categories)):
            for prob in range(len(self.categories[var])):
                if prob != sample[var]:
                    self.global_best[var][prob] = self.scaling_factor * position[var][prob]
                else:
                    summation = sum([(1.0 - self.scaling_factor) * pr if position[var].index(pr) != prob else 0 for pr in position[var]])
                    self.global_best[var][prob] = position[var][prob] + summation

    def update_local_best(self, particle, position, sample):
        for var in range(len(self.categories)):
            for prob in range(len(self.categories[var])):
                if prob != sample[var]:
                    self.local_best[particle][var][prob] = self.scaling_factor * position[var][prob]
                else:
                    summation = sum([(1.0 - self.scaling_factor) * pr if position[var].index(pr) != prob else 0 for pr in position[var]])
                    self.local_best[particle][var][prob] = position[var][prob] + summation

    def sample_distribution(self, distribution):
        # print(self.categories[0], distribution[0])
        sample = [np.random.choice(self.categories[var], p=distribution[var]) for var in range(len(distribution))]
        return sample

    def calculate_new_velocities(self):
        print(self.velocities.shape)
        self.inertia_weight * self.velocities
        self.velocities = self.inertia_weight * self.velocities + \
                          self.weight_global * random() * ([self.global_best for _ in range(len(self.positions))] - self.positions) + \
                          self.weight_local * random() * (self.local_best - self.positions)

    def run(self):
        self.initialise_categorical_positions()
        self.initialise_categorical_velocities()
        print(self.velocities)
        for iteration in range(self.n_iterations):
            # print(iteration)
            self.samples = [self.representative_sample(position) for position in self.positions]
            self.fitness = [self.fitness_function(position) for position in self.positions]

            for particle in range(self.n_particles):
                if self.fitness[particle] < self.local_best_fitness[particle]:
                    self.update_local_best(particle, self.positions[particle], self.samples[particle])

            max_local = max(self.local_best_fitness)
            if max_local < self.global_best_fitness:
                self.update_global_best(self.local_best.index(max_local), self.representative_sample(self.local_best.index(max_local)))

            self.calculate_new_velocities()
            self.update_position()
        print(self.global_best_fitness, self.global_best)

if __name__ == '__main__':
    # print(generateWeights(3))
    # n = 10000
    # weight1 = np.zeros((n, 3))
    # for i in range(n):
    #     weight1[i] = generateWeights(3)
    #
    # plt.figure()
    # plt.scatter(weight1[:, 0], weight1[:, 1], s=0.3)
    # # ax = plt.gca(projection='3d')
    # # ax.scatter(weight1[:, 0], weight1[:, 1], weight1[:, 2], s=0.3)
    # plt.show()
    opt = PSOCategorical(function1, 5, 0, 0.5)
    opt.run()
