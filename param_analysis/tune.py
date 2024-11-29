
import json
import argparse
import sys
import multiprocessing
import numpy as np
import time
import itertools
from collections import defaultdict

sys.path.append("/home/ale234/github/BreedingProgram")
from BreedingProgram import run_breeder, extract_logic  # noqa


def group_by_key(dict_list, group_keys, avg=True):
    # groups dictionaries in dict_list with same group_keys
    # if avg the other keys are averaged else they are set as lists
    grouped = defaultdict(lambda: defaultdict(list))

    for d in dict_list:
        group_values = tuple(d[key] for key in group_keys)
        for key, value in d.items():
            if key in group_keys:
                grouped[group_values][key] = value
            else:
                grouped[group_values][key].append(value)

    result = []
    for group_values, combined_dict in grouped.items():
        processed_dict = {key: value for key, value in combined_dict.items()}
        for key, values in combined_dict.items():
            if key not in group_keys:
                if avg:
                    processed_dict[key] = np.mean(values)  # Compute average
                else:
                    processed_dict[key] = values  # Collect into a list
        result.append(processed_dict)

    return result


def print_progress(n, total, start_time):
    percent = 100 * (n / total)
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / \
        (n + 1) * total
    eta = estimated_total_time - elapsed_time
    min = np.floor(eta / 60)
    sec = eta % 60
    eta = f"{min:.0f} m {sec:.0f} s  " if min > 0 else f"{sec:.0f} s  "
    print(f"\r\tEvaluated {n} out of {
          total}: {percent:.2f}% - ETA:  " + eta, end='')


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run parallel processing with specified CPU cores and output results to a file.")
    parser.add_argument(
        "--cpus",
        type=int,
        required=True,
        help="Number of CPU cores to use for multiprocessing."
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Path to the output JSON file where results will be saved."
    )
    args = parser.parse_args()

    cpus = args.cpus
    out_file = args.o

    # 1: define combinations of settings

    # TOURNAMENT TUNING
    pop_size = [100, 250, 450]
    n_genes = [15, 25]
    sel = ["tournament"]
    cross = ["1p", "2p"]
    mut = ["flip", "swap"]
    ps = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pc = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    pm = [0.05, 0.06, 0.07, 0.08, 0.09, 1, 1.1]
    max_gen = [300]
    T0 = [0]
    alpha = [0]

    # ENTROPY TUNING
    # pop_size = [200, 340, 500]
    # n_genes = [15, 25]
    # sel = ["entropy"]
    # cross = ["1p", "2p"]
    # mut = ["flip", "swap"]
    # ps = [0.2, 0.3, 0.4, 0.5]
    # pc = [0.6, 0.7, 0.8, 0.9]
    # pm = [0.03, 0.05, 0.07, 0.09]
    # max_gen = [350]
    # T0 = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    # alpha = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # each perm will be input to run_breeder()
    perm = list(itertools.product(
        *[pop_size, n_genes, sel, cross, mut, ps, pc, pm, T0, alpha, max_gen]))

    # last batch size if len(perm) is not divisible by cpus
    rest = len(perm) % cpus

    # 2: evaluate all combinations with multiprocessing
    best = []
    print(f"\n\tCpus = {cpus}")
    # Main loop
    with multiprocessing.Pool(cpus) as pool:
        start_time = time.time()
        for n in range(cpus, len(perm), cpus):
            batch = perm[n - cpus: n]
            res = pool.map(run_breeder, batch)
            best.extend(res)
            print_progress(n, len(perm), start_time)

    # Last batch
    if rest > 0:
        with multiprocessing.Pool(rest) as pool:
            batch = perm[-rest:]
            res = pool.map(run_breeder, batch)
            best.extend(res)
            print_progress(len(perm), len(perm), start_time)
    else:
        with multiprocessing.Pool(cpus) as pool:
            batch = perm[-cpus:]
            res = pool.map(run_breeder, batch)
            best.extend(res)
            print_progress(len(perm), len(perm), start_time)

    # 3: select promising results
    id = [best[i].fitness < -78.3322 for i in range(len(best))]
    perm = extract_logic(perm, id)
    promising = extract_logic(best, id)

    # 4: re-test them, only if the result will be correct N times they will be accepted
    N = 5

    optimal = []

    print(f"\n\tRe-testing {len(perm)} promising configurations {N} times...")
    if N <= cpus:
        with multiprocessing.Pool(N) as pool:
            start_time = time.time()
            for n in range(len(perm)):
                res = pool.map(run_breeder, [perm[n]] * N)
                if all([r.fitness < -78.3322 for r in res]):
                    optimal.append(perm[n])
                print_progress(n, len(perm), start_time)
    else:
        rest = N % cpus
        with multiprocessing.Pool() as pool:
            start_time = time.time()
            for n in range(len(perm)):
                res = []
                for i in range(0, N - rest, cpus):
                    res.append(pool.map(run_breeder, [perm[n]]*cpus))
                if rest > 0:
                    res.append(pool.map(run_breeder, [perm[n]]*rest))
                if all([r.fitness < -78.3322 for r in res]):
                    optimal.append(perm[n])
                print_progress(n, len(perm), start_time)

    # 5: log data into output JSON file
    dict_lst = []
    for opt in optimal:
        dictionary = {
            "pop_size": opt[0],
            "n_genes": opt[1],
            "sel": opt[2],
            "cross": opt[3],
            "mut": opt[4],
            "ps": opt[5],
            "pc": opt[6],
            "pm": opt[7],
            "T0": opt[8],
            "alpha": opt[9],
            "max_gen": opt[10]
        }
        dict_lst.append(dictionary)

    # group dictionaries with same sel, cross and mut
    dict_lst = group_by_key(
        dict_lst,
        ["pop_size", "n_genes", "sel", "cross", "mut"],
        avg=True
    )
    dict_lst = group_by_key(
        dict_lst,
        ["sel", "cross", "mut"],
        avg=False
    )

    with open(out_file, "w", newline="") as of:
        of.write(json.dumps(dict_lst, indent=4))
    print("\n\tDone. Optimal combinations logged into output file\n")
