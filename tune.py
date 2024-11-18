from GeneticAlgorithm import BreedingProgram
import multiprocessing
import time
import itertools
import numpy as np

def get_permutations(vectors):
    all_permutations = list(itertools.product(*vectors))
    return np.array(all_permutations)

def Styblinski_Tang(x):
     return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323

def self_tune_boltzmann(vars):
    T0, alpha = vars

    bp = BreedingProgram(problem_size=2, pop_size=100, n_genes=25)
    bp.problem_type = "Minimize"
    bp.selection_method = "Boltzmann"
    bp.crossover_method = "one point"
    bp.ps = 0.25
    bp.pc = 0.88
    bp.pm = 0.08

    bp.T0 = T0
    bp.alpha = alpha

    search_space = [[-5,5],[-5,5]]
    bp.start_evolution(Styblinski_Tang, search_space, eps=1e-9, log=False, plot=False)

    return bp.best[-1]

def self_tune_tournament(vars):
    ps, pc, pm = vars

    bp = BreedingProgram(problem_size=2, pop_size=100, n_genes=25)
    bp.problem_type = "Minimize"
    bp.selection_method = "Tournament"
    bp.crossover_method = "one point"
    bp.ps = ps
    bp.pc = pc
    bp.pm = pm
    search_space = [[-5,5],[-5,5]]
    bp.start_evolution(Styblinski_Tang, search_space, eps=1e-9, max_gen=50, log=False, plot=False)

    return bp.best[-1]



if __name__ == "__main__":
     
  
    cpus = multiprocessing.cpu_count()

    n_max = 5*cpus # keep multiple of cpus

    # T0 = np.linspace(1, 7, n_max)
    # alpha = np.linspace(0.01, 0.99, n_max) 

    ps = np.linspace(0.01, 0.99, n_max)
    pc = np.linspace(0.01, 0.99, n_max)
    pm = np.linspace(0.01, 0.3, n_max)
    perm = get_permutations([ps, pc, pm]) 

    best = []
    params = []
    start_time = time.time()
    for i in range(0,len(perm), cpus):
        with multiprocessing.Pool() as pool:
            iterable = [ perm[k,:] for k in range(i, i+cpus) ]
            result = pool.map(self_tune_tournament, iterable)

        best.append( result[np.argmin([ind.fitness for ind in result])] )
        params.append(perm[i,:])
        if i > 0:
            print(f"\niteration {int(i/cpus)} of {int(len(perm)/cpus)}, {best[-1].decoded_genome}, fit = {best[-1].fitness}", end='')
    stop_time = time.time()

    print(f"\n\nTuning time with {cpus} cpus: {stop_time-start_time} s")
    print("Best result: ")
    best_id = np.argmin([ind.fitness for ind in best])
    best[best_id].display_genome(decoded=True)
    print(f"With parameters: {params[best_id]}")

    n_test = 5
    print("\nTesting optimal parameters {n_test} times")
    for i in range(n_test):
        out = self_tune_boltzmann(params[best_id])
        out.display_genome()