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
    return 

search_space = [[-1,1], [-1,1]]

bp = BreedingProgram(2)
bp.problem_type = "Minimize"
bp.create_pop(20, 2, 10, log=True)
bp.display_pop()