import numpy as np
from GeneticAlgorithm import BreedingProgram, extract_logic

def func(vars):
    x,y = vars
    return -x**2 -y**2

bp = BreedingProgram(2)
bp.problem_type = "Maximize"
search_space = [[-1,1], [-1,1]]
bp.create_pop(1000, 2, 50, log=True)

bp.evaluate_finess(func, search_space, rank=True, log=True)
print(f"Avg fitness: {bp.fitness.mean()}")
bp.select("Boltzman", 0.2, "Minimize", log=True)
bp.breed("one point", 0.9, log=True)


