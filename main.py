from GeneticAlgorithm import BreedingProgram
import numpy as np

def print_step(pop, idx):
    if len(pop) != len(idx):
        print("Noooo")
        return
    for i in range(len(pop)):
        print(f"{idx[i]}  {pop[i][0]}{pop[i][1]}")
    print("\n\n")

def func3d(vars):
    x,y,z = vars
    return 5*x**2-4*y**2+3*z**2-2*x*y+y*z

def Styblinski_Tang(x):
     return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2

def main():

    search_space = [[-5,1], [-5,1]]
    bp = BreedingProgram(2)

    bp.pop_size = 100
    bp.n_genes = 25
    bp.problem_type = "Minimize"
    bp.selection_method = "Tournament"
    bp.T0 = 10000
    bp.ps = 0.2
    bp.pc = 0.9
    bp.pm = 0.1
    
    bp.start_evolution(Styblinski_Tang, search_space, max_gen = 1000, eps = 1e-9)


    
    
    


if __name__ == "__main__":
    main()

