from BreedingProgram import Breeder
import numpy as np


def func1d(x):
    return -abs(np.cos(x) - 0.25*x)


def func3d(x):
    return np.sin(x[0])*np.cos(x[1])*np.exp(-x[2]**2) + x[0]**2 - x[1]**2 + x[2]
# Maximum @ x = (6.283185307179586, 0, 2.0),  f = 41.47841760435743


def Styblinski_Tang(x):
    return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


if __name__ == "__main__":

    # 1d function search space
    # search_space = ( (1, 3) )

    # 3d function search space
    # search_space = ((-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-2, 2))

    # Styblinski_Tang search space
    search_space =((-5, 5), (-5, 5))

    sol = (-2.90353, -2.90353, -78.3323)

    breeder = Breeder(problem_size=2)
    breeder.pop_size = 250
    breeder.n_genes = 25
    breeder.problem_type = "minimize"
    breeder.selection_method = "tournament"
    breeder.T0 = 3.6
    breeder.alpha = 0.55
    breeder.crossover_method = "2p"
    breeder.mutation_method = "flip"
    breeder.ps = 0.4954982146671957
    breeder.pc = 0.5606836146122623
    breeder.pm = 0.09525101474044007
    breeder.start_evolution(Styblinski_Tang, search_space, cpus=1, max_gen=250, eps=1e-9, log=True, plot=True, sol=sol)
