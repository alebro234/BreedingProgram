import numpy as np
import GeneticAlgorithm as ga
import matplotlib.pyplot as plt

f_max = 78
f_min = 35
problem_type = "Maximize"
pop_size = 100

f = np.linspace(f_min, 2*f_max, pop_size)
delta_f = abs(f - f_max)
kb = 1.380649e-23
for T in [1, 10, 20]:
    P = np.exp( -delta_f/T ) / np.exp( -delta_f/T ).sum()
    plt.plot(f, P)
plt.legen()
plt.show()