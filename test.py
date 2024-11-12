import numpy as np
from GeneticAlgorithm import BreedingProgram

def func(vars):
    x,y = vars
    return x**2 +y**2

bp = BreedingProgram(2)
bp.problem_type = "Minimize"
search_space = [[-1,1], [-1,1]]
bp.create_pop(10, 2, 5, log=True)

for i in range(3):
    bp.evaluate_finess(func, search_space, rank=False, log=True)
    bp.select("Tournament", 0.2, "Minimize", log=True)
    bp.breed("one point", 0.9, log=True)
