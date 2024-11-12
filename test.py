import numpy as np
from GeneticAlgorithm import Population, BreedingProgram


def print_step(pop, idx):
    if len(pop) != len(idx):
        print("Noooo")
        return
    for i in range(len(pop)):
        print(f"{idx[i]}  {[chrom for chrom in pop[i]]}")
    print("\n\n")


def func(x):
    return x**2

bp = BreedingProgram(1)

bp.create_pop(10, 1, 10, T0=10)
bp.evaluate_finess(func, [[-1,1]])
print_step(bp.pop, np.arange(bp.pop_size))
curr, worthy, id = bp.select("Boltzman", 1, "Minimize")
print(worthy)
print(id)




