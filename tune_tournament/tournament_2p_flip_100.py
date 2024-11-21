import os
import csv
import numpy as np
import itertools
import time
import multiprocessing
import sys

sys.path.append("/home/ale234/github/BreedingProgram")
from GeneticAlgorithm import BreedingProgram, extract_id  # noqa


def print_progress(n, total, start_time):
    percent = 100 * (n / total)
    bar = '█' * int(percent // 5) + '-' * (20 - int(percent // 5))
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / \
        (n + 1) * total  # Estimate total time
    eta = estimated_total_time - elapsed_time  # Remaining time
    min = np.floor(eta / 60)
    sec = eta % 60
    eta = f"{min:.0f} m {sec:.0f} s  " if min > 0 else f"{sec:.0f} s  "
    print(f"\rEvaluated {n} out of {
          total}: |{bar}| {percent:.2f}% - ETA:  " + eta, end='')


def get_permutations(vectors):
    all_permutations = list(itertools.product(*vectors))
    return np.array(all_permutations)


def Styblinski_Tang(x):
    return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.9035, -2.9035), f = -78.3323


def tune_entropy(vars):
    ps, pc, pm, T0, alpha = vars
    bp = BreedingProgram(problem_size=2, pop_size=100, n_genes=25)
    bp.problem_type = "minimize"
    bp.selection_method = "entropy"
    bp.crossover_method = "two-point"
    bp.mutation_scheme = "bit-flip"
    bp.ps = ps
    bp.pc = pc
    bp.pm = pm
    bp.T0 = T0
    bp.alpha = alpha

    search_space = [[-5, 5], [-5, 5]]
    bp.start_evolution(Styblinski_Tang, search_space,
                       max_gen=500, eps=1e-9, log=False, plot=False)

    return bp.best[-1]


def tune_tournament(vars):
    ps, pc, pm = vars

    bp = BreedingProgram(problem_size=2, pop_size=100, n_genes=25)
    bp.problem_type = "minimize"
    bp.selection_method = "tournament"
    bp.crossover_method = "one-point"
    bp.ps = ps
    bp.pc = pc
    bp.pm = pm
    search_space = [[-5, 5], [-5, 5]]
    bp.start_evolution(Styblinski_Tang, search_space,
                       eps=1e-9, max_gen=100, log=False, plot=False)

    return bp.best[-1]


if __name__ == "__main__":

    file_name = os.path.basename(__file__)
    print("\n\nRunning " + file_name)

    # use all (8)
    cpus = multiprocessing.cpu_count()

    # define parameters vectors
    ps = np.linspace(0.1, 0.5, cpus)
    pc = np.linspace(0.7, 0.99, cpus)
    pm = np.linspace(0.01, 1.5, cpus)
    T0 = np.linspace(1, 5, cpus)
    alpha = np.linspace(0.1, 0.99, cpus)

    # len(perm) will be cpus**m where m number of parameters to tune
    perm = get_permutations([ps, pc, pm, T0, alpha])
    # store results
    best = []
    params = []

    print(f"\n\tTuning {len(perm)} sets of parameters\n\tCpus = {cpus}")
    with multiprocessing.Pool(cpus) as pool:
        start_time = time.time()
        for n in range(0, len(perm), cpus):
            iterable = [perm[k, :] for k in range(n, n+cpus)]
            result = pool.map(tune_entropy, iterable)

            best.extend([ind for ind in result])
            print_progress(n, len(perm), start_time)

    # rank individuals
    ranked_id = np.argsort([ind.fitness for ind in best])

    best = extract_id(best, ranked_id)
    perm = perm[ranked_id]

    # log n_best individuals to output csv file
    n_best = 20
    data = [["ps", "pc", "pm", "T0", "alpha", "x", "y", "fit"]]
    print(f"\n\n{n_best} best individuals")
    print(data)
    for n in range(n_best):
        data_row = []
        data_row.extend(perm[n, :].tolist())
        data_row.extend(best[n].decoded_genome)
        data_row.append(best[n].fitness)
        data.append(data_row)
        print(f"{n}", data_row)

    with open(file_name + "csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
