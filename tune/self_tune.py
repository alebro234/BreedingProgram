# Run the GA to tune its parameters for the ND Styblinsky-Tang function
# USAGE
# python self_tune.py --cpus 4 -i input_file.json -o output_file.json -N 2



import sys
import os
import json
import argparse
import numpy as np
import multiprocessing

sys.path.append("/home/ale234/github/BreedingProgram")
from BreedingProgram import Breeder, Styblinski_Tang  # noqa

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("-i", type=str, required=False)
    parser.add_argument("-o", type=str, required=True)
    parser.add_argument("-N", type=int, required=False, default=2)

    args = parser.parse_args()
    cpus = min(args.cpus, multiprocessing.cpu_count())
    input_file = args.i
    out_file = args.o

    # Tuning settings if no input file is provided
    if input_file is None:
        settings_lst = [
        {
            "sel": "entropy",
            "cross": "2p",
            "mut": "swap",
            "pop_size": 250,
            "n_genes": 25,
            "max_gen": 250,
            "ps_lim": [0.1, 0.8],
            "pc_lim": [0.5, 0.99],
            "pm_lim": [0.01, 0.15],
            "T0_lim": [1, 10],
            "alpha_lim": [0.1, 0.95]
        }
        ]
    else:
        with open(input_file, "r") as f:
            settings_lst = json.load(f)


    # dimension of Styblinski-Tang function for which to tune
    test_DIM = args.N
    test_search_space = []
    for _ in range(test_DIM):
        test_search_space.append((-5, 5))


    out_file = "new_1d_optimized.json"

    os.system("rm -f " + out_file)
    with open(out_file, "w", newline="") as f:
        json.dump([], f)

    for settings in settings_lst:

        # define error function, problem size and search space
        
        if settings["sel"] == "tournament":

            def error_func(vars):
                ps, pc, pm = vars

                breeder = Breeder(problem_size=test_DIM)
                breeder.pop_size = settings["pop_size"]
                breeder.n_genes = settings["n_genes"]
                breeder.selection_method = "tournament"
                breeder.crossover_method = settings["cross"]
                breeder.mutation_method = settings["mut"]
                breeder.ps = ps
                breeder.pc = pc
                breeder.pm = pm

                breeder.start_evolution(Styblinski_Tang, test_search_space, max_gen=settings["max_gen"], log=False, plot=False)

                errnorm = [abs(1 - breeder.best[-1].fitness/(-78.33234))]
                errnorm.extend([abs(1 - x/(-2.903534)) for x in breeder.best[-1].decoded_genome])
                errnorm = 100 * np.linalg.norm(errnorm)

                return errnorm

            N = 3
            search_space = ( settings["ps_lim"], settings["pc_lim"], settings["pm_lim"] )

        elif settings["sel"] == "entropy":

            def error_func(vars):
                ps, pc, pm, T0, alpha = vars

                breeder = Breeder(problem_size=test_DIM)
                breeder.pop_size = settings["pop_size"]
                breeder.n_genes = settings["n_genes"]
                breeder.selection_method = "entropy"
                breeder.crossover_method = settings["cross"]
                breeder.mutation_method = settings["mut"]
                breeder.ps = ps
                breeder.pc = pc
                breeder.pm = pm
                breeder.T0 = T0
                breeder.alpha = alpha

                breeder.start_evolution(Styblinski_Tang, test_search_space, max_gen=settings["max_gen"], log=False, plot=False)

                errnorm = [abs(1 - breeder.best[-1].fitness/(-78.33234))]
                errnorm.extend([abs(1 - x/(-2.903534)) for x in breeder.best[-1].decoded_genome])
                errnorm = 100 * np.linalg.norm(errnorm)
            
                return errnorm

            N = 5
            search_space = ( settings["ps_lim"], settings["pc_lim"], settings["pm_lim"], settings["T0_lim"], settings["alpha_lim"] )

        else:
            raise ValueError("Invalid selection method")





        # let the self tuning start

        breeder = Breeder(problem_size=N, problem_type="minimize")
        breeder.selection_method = "entropy"
        breeder.crossover_method = "2p"
        breeder.mutation_method = "flip"
        breeder.pop_size = 200
        breeder.n_genes = 25
        breeder.ps = 0.3
        breeder.pc = 0.9
        breeder.pm = 0.1
        breeder.T0 = 7
        breeder.alpha = 0.75

        breeder.start_evolution(error_func, search_space, eps=1e-9, cpus=cpus, max_gen=250, plot=False)

        output = {
            "sel": settings["sel"],
            "cross": settings["cross"],
            "mut": settings["mut"],
            "pop_size": settings["pop_size"],
            "n_genes": settings["n_genes"],
            "max_gen": settings["max_gen"],
            "ps": breeder.best[-1].decoded_genome[0],
            "pc": breeder.best[-1].decoded_genome[1],
            "pm": breeder.best[-1].decoded_genome[2],
        }
        if settings["sel"] == "entropy":
            output["T0"] = breeder.best[-1].decoded_genome[3]
            output["alpha"] = breeder.best[-1].decoded_genome[4]


        # log each results when done to output JSON file
        

        with open(out_file, "r") as f:
            prev_output = json.load(f)
        prev_output.append(output)
        
        with open(out_file, "w", newline="") as f:
            json.dump(prev_output, f, indent=4)

