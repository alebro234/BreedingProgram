from GeneticAlgorithm import BreedingProgram
import numpy as np

def func2d(x):
    return 0.2*x[1]**2 + 0.2*x[0]**2 + 2*np.sin(x[0])

def func3d(x):
    return np.sin(x[0]) * np.cos(x[1]) * np.exp(-x[2]**2) + x[0]**2 - x[1]**2 + x[2]
# Maximum @ x = (2*pi, 0, 2.0),  f = 41.47841760435743

def Styblinski_Tang(x):
     return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


def main():

    search_space = [[-2*np.pi,2*np.pi], [-2*np.pi,2*np.pi], [-2,2]]
    # search_space = [[-5.5,5.5],[-5.5,5.5]]
    bp = BreedingProgram(3, pop_size=1000, n_genes=25)

    bp.problem_type = "Maximize"
    bp.selection_method = "Tournament"
    bp.crossover_method = "two point"
    bp.T0 = 3.7
    bp.alpha = 0.7379661
    bp.ps = 0.25
    bp.pc = 0.93
    bp.pm = 0.1
    
    bp.start_evolution(func3d, search_space, eps = 1e-9, log=True, plot=True)
    
    


if __name__ == "__main__":
    main()

