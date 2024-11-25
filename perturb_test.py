import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
from GeneticAlgorithm import run_breeder


def apply_perturb(settings, amplitude, id):
    # amplitude: +- percent to apply
    ptb = 1 + np.random.uniform(-amplitude, +amplitude)
    for key in ["ps", "pc", "pm", "T0", "alpha"]:
        settings[key][id] *= ptb
    return settings, ptb


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
        help="Input JSON file"
    )

    args = parser.parse_args()
    in_file = args.input

    with open(in_file, "r") as Ifile:
        dict_lst = json.load(Ifile)

    # order dict keys for conversion to list

    MaxAmplitudes = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

    ptb_ampl = []
    err = []
    cpus = multiprocessing.cpu_count()

    with multiprocessing.Pool(cpus) as pool:
        for settings in dict_lst:
            sel, cross, mut = settings["sel"], settings["cross"], settings["mut"]
            plt.figure()

            # run through all combinations of pop_size and n_genes
            for n in range(len(settings["pop_size"])):
                pop_size = settings["pop_size"][n]
                n_genes = settings["n_genes"][n]
                print(f"\n\tPerturbing optimal {sel}, {cross}, {
                      mut}, pop_size = {pop_size}, n_genes = {n_genes}")
                local_ptb_ampl = []
                local_fit = []

                for A in MaxAmplitudes:
                    print(f"\tmax|A| = {100 * A} % ")
                    nth_settings = [settings]
                    perturbations = [0]

                    for _ in range(cpus-1):
                        perturbed_settings, perturbation = apply_perturb(
                            nth_settings[0], A, n)
                        perturbations.append(100 * (1 - perturbation))
                        nth_settings.append(perturbed_settings)

                    # Parallelize breeder runs
                    nth_settings = [settings_to_input(s, n)
                                    for s in nth_settings]
                    res = pool.map(run_breeder, nth_settings)
                    local_fit.extend([best.fitness for best in res])
                    local_ptb_ampl.extend(perturbations)

                err.extend(local_fit)
                ptb_ampl.extend(local_ptb_ampl)

                plt.scatter(local_ptb_ampl, local_fit, label=f"pop_size={
                            pop_size}, n_genes={n_genes}")
                print("\tDone.")

            # Create and customize the plot
            plt.xlabel("Perturbation Amplitude (A %)")
            plt.ylabel("Fitness")
            plt.title(f"{sel},{cross},{mut}")
            # plt.ylim(-78.3323, -78)
            plt.legend()
    plt.show()
