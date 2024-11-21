import numpy as np
import matplotlib.pyplot as plt
from GeneticAlgorithm import BreedingProgram
import csv


def Styblinski_Tang(x):
    return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


def run_breeder(pop_size, n_genes, sel, cross, mut, ps, pc, pm, T0, alpha):

    bp = BreedingProgram(2, "minimize", pop_size, n_genes)
    bp.selection_method = sel
    bp.crossover_method = cross
    bp.mutation_scheme = mut
    bp.ps = ps
    bp.pc = pc
    bp.pm = pm
    bp.T0 = T0
    bp.alpha = alpha

    bp.start_evolution(Styblinski_Tang, [
                       [-5, 5], [-5, 5]], log=False, plot=False)

    return bp.best[-1]


if __name__ == "__main__":

    pass
