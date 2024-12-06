# script to run the ND Styblinsky-Tang minimization with the settings provided in an input JSON file
# USAGE
# python run_ga.py -i input_file.json -N 2

import sys
import json
import argparse
sys.path.append("/home/ale234/github/BreedingProgram")
from BreedingProgram import run_breeder_test  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True)
    parser.add_argument("-N", type=int, required=False, default=2)
    args = parser.parse_args()
    input_file = args.i
    N = args.N

    with open(input_file, "r") as f:
        settings_lst = json.load(f)

    for settings in settings_lst:

        print(f"Running GA in {N}D with settings:")
        for key, item in settings.items():
            print(f"{key}: {item}")

        out = run_breeder_test(settings, N)

        out["best"].display_genome(decoded=True)
        print(f"errnorm {out["errnorm"]}% , sim time: {out["sim_time"]:.2f} s")
        print("\n\n")
