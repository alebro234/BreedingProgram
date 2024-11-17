from GeneticAlgorithm import BreedingProgram
import numpy as np

def func2d(x):
    return 0.2*x[1]**2 + 0.2*x[0]**2 + 2*np.sin(x[0])

def func3d(x):
    return np.sin(x[0]) * np.cos(x[1]) * np.exp(-x[2]**2) + x[0]**2 - x[1]**2 + x[2]
# Maximum @ x = (6.283185307179586, 0, 2.0),  f = 41.47841760435743

def Styblinski_Tang(x):
     return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


def main():

    # search_space = [[-2*np.pi,2*np.pi], [-2*np.pi,2*np.pi], [-2,2]]
    search_space = [[-5.5,5.5],[-5.5,5.5]]
    bp = BreedingProgram(2, pop_size=1000, n_genes=25)

    bp.problem_type = "Minimize"
    bp.selection_method = "Boltzmann"
    bp.crossover_method = "one point"
    bp.T0 = 2.9
    bp.alpha = 0.7379661
    bp.ps = 0.25
    bp.pc = 0.93
    bp.pm = 0.06
    
    bp.start_evolution(Styblinski_Tang, search_space, eps = 1e-12, log=True, plot=True)
    
    


if __name__ == "__main__":
    main()

