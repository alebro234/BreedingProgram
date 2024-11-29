import sys
import json
import argparse
import numpy as np
import multiprocessing

sys.path.append("/home/ale234/github/BreedingProgram")
from BreedingProgram import Breeder  # noqa

def Styblinski_Tang(x):
    return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2
# Minimum @ x = (-2.90354, -2.90354), f = -78.3323


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--input", type=str, required=False)
    parser.add_argument("--mode", type=str, default="a")

    args = parser.parse_args()
    cpus = min(args.cpus, multiprocessing.cpu_count())
    input_file = args.input 
    mode = args.mode

    # Tuning settings if no input file is provided
    if input_file is None:
        settings_lst = [{
            "sel": "entropy",
            "cross": "1p",
            "mut": "swap",
            "pop_size": 250,
            "n_genes": 25,
            "max_gen": 500,
            "ps_lim": [0.1, 0.99],
            "pc_lim": [0.5, 0.99],
            "pm_lim": [0.01, 0.15],
            "T0_lim": [1, 10],
            "alpha_lim": [0.1, 0.95],
        }]
    else:
        with open(input_file, "r") as f:
            settings_lst = json.load(f)


    # iterate over settings

    output = []

    for settings in settings_lst:

        # define error function, problem size and search space
        
        if settings["sel"] == "tournament":

            def error_func(vars):
                ps, pc, pm = vars

                breeder = Breeder(problem_size=2)
                breeder.pop_size = settings["pop_size"]
                breeder.n_genes = settings["n_genes"]
                breeder.selection_method = "tournament"
                breeder.crossover_method = settings["cross"]
                breeder.mutation_method = settings["mut"]
                breeder.ps = ps
                breeder.pc = pc
                breeder.pm = pm

                breeder.start_evolution(Styblinski_Tang, ((-5, 5), (-5, 5)), max_gen=settings["max_gen"], log=False, plot=False)

                err = [abs(-78.3323 - breeder.best[-1].fitness)]
                err.extend([abs(-2.90354 - x) for x in breeder.best[-1].decoded_genome])
                err = np.linalg.norm(err)

                return err

            N = 3
            search_space = ( settings["ps_lim"], settings["pc_lim"], settings["pm_lim"] )

        elif settings["sel"] == "entropy":

            def error_func(vars):
                ps, pc, pm, T0, alpha = vars

                breeder = Breeder(problem_size=2)
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

                breeder.start_evolution(Styblinski_Tang, ((-5, 5), (-5, 5)), max_gen=settings["max_gen"], log=False, plot=False)

                err = [abs(-78.3323 - breeder.best[-1].fitness)]
                err.extend([abs(-2.90354 - x) for x in breeder.best[-1].decoded_genome])
                err = np.linalg.norm(err)
                

                return err

            N = 5
            search_space = ( settings["ps_lim"], settings["pc_lim"], settings["pm_lim"], settings["T0_lim"], settings["alpha_lim"] )

        else:
            raise ValueError("Invalid selection method")

        # let the self tuning start

        breeder = Breeder(problem_size=N, problem_type="minimize")
        breeder.selection_method = "entropy"
        breeder.crossover_method = "1p"
        breeder.mutation_method = "flip"
        breeder.pop_size = 250
        breeder.n_genes = 15
        breeder.ps = 0.2
        breeder.pc = 0.8
        breeder.pm = 0.1
        breeder.T0 = 3.6
        breeder.alpha = 0.55

        breeder.start_evolution(error_func, search_space, eps=1e-9, cpus=cpus, max_gen=1000, plot=False)

        result = {
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
            result["T0"] = breeder.best[-1].decoded_genome[3]
            result["alpha"] = breeder.best[-1].decoded_genome[4]

        output.append(result)

    # log results to output JSON file
    
    out_file = "optimized.json"

    if mode == "a":
        with open(out_file, "r") as f:
            prev_output = json.load(f)
        prev_output.extend(output)
         
        with open(out_file, "w", newline="") as f:
            json.dump(prev_output, f, indent=4)
    else:
        with open(out_file, "w", newline="") as f:
            json.dump(output, f, indent=4)

