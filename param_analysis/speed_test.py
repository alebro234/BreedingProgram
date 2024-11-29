import numpy as np
import json
import time
import argparse
import sys

sys.path.append("/home/ale234/github/BreedingProgram")
from BreedingProgram import run_breeder   # noqa


def settings_to_input(dictionary, id):
    # turns dictionary settings into run_breeder input list
    keys = ["pop_size", "n_genes", "sel", "cross",
            "mut", "ps", "pc", "pm", "T0", "alpha", "max_gen"]
    return [
        value[id] if isinstance(value, list) else value
        for key in keys if key in dictionary
        for value in [dictionary[key]]
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run perturbative analysis of optimized parameters"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with optimized settings"
    )

    args = parser.parse_args()
    in_file = args.input

    # load optimized settings
    with open(in_file, "r") as Ifile:
        dict_list = json.load(Ifile)

    # number of tests for each setting
    N = 3

    for settings in dict_list:
        # run through all combinations of pop_size and n_genes
        for i in range(len(settings["pop_size"])):
            sel = settings["sel"]
            cross = settings["cross"]
            mut = settings["mut"]
            pop_size = settings["pop_size"][i]
            n_genes = settings["n_genes"][i]
            print(f"\n\tTesting {sel},{cross},{mut},pop size = {
                  pop_size}, n_genes = {n_genes} {N} times")
            avg_time = []
            avg_fit = []
            for n in range(N):
                start_time = time.time()
                input_settings = settings_to_input(settings, i)
                best = run_breeder(input_settings)
                stop_time = time.time()
                avg_time.append(stop_time - start_time)
                avg_fit.append(best.fitness)
            avg_time = np.mean(avg_time)
            avg_fit = np.mean(avg_fit)
            print(f"\tAvg time = {
                  avg_time:.4f} s, Avg fitness = {avg_fit:.5f}")
