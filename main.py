from GeneticAlgorithm import BreedingProgram
import numpy as np


def func2d(x):
    return 0.2*x[1]**2 + 0.2*x[0]**2 + 2*np.sin(x[0])


def func3d(x):
    return np.sin(x[0])*np.cos(x[1])*np.exp(-x[2]**2) + x[0]**2 - x[1]**2 + x[2]
# Maximum @ x = (6.283185307179586, 0, 2.0),  f = 41.47841760435743


def Styblinski_Tang(x):
    return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


def main():
    # 3d function search space
    # search_space = [[-2*np.pi,2*np.pi], [-2*np.pi,2*np.pi], [-2,2]]

    # Styblinski_Tang search space
    search_space = [[-5, 5], [-5, 5]]

    bp = BreedingProgram(problem_size=2)
    bp.pop_size = 500
    bp.n_genes = 25
    bp.problem_type = "minimize"
    bp.selection_method = "entropy"
    bp.T0 = 2.6
    bp.alpha = 0.75
    bp.crossover_method = "two-point"
    bp.mutation_scheme = "bit-swap"
    bp.ps = 0.3
    bp.pc = 0.9
    bp.pm = 0.08
    bp.start_evolution(Styblinski_Tang, search_space,
                       eps=1e-12, log=True, plot=True)


if __name__ == "__main__":
    main()
