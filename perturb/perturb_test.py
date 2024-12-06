# Run perturbative analysis of the input settings one the ND Styblinsky-Tang function
# USAGE
# python perturb_test.py --cpus 4 -i input_file.json -o output_file.json -N 2



import numpy as np
import argparse
import json
import multiprocessing
import sys

sys.path.append("/home/ale234/github/BreedingProgram")
from BreedingProgram import run_breeder_test  # noqa


def perturb_and_run(settings, A, N):
    for key in ["ps", "pc", "pm", "T0", "alpha"]:
        if key in settings.keys():
            ptb = np.random.uniform(0, A)
            settings[key] *= 1 + ptb

    return run_breeder_test(settings, N)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",type=str,required=True)
    parser.add_argument("-o",type=str,required=True)
    parser.add_argument("--cpus",type=int,required=True)
    parser.add_argument("-N",type=int,required=False,default=2)

    args = parser.parse_args()
    in_file = args.i
    cpus = min(args.cpus, multiprocessing.cpu_count())
    out_file = args.o
    N = args.N

    with open(in_file, "r") as Ifile:
        settings_lst = json.load(Ifile)

    # Define perturbation amplitudes
    amplitudes = np.linspace(-0.1, 0.1, 150)
    
    done = 0
    with multiprocessing.Pool(cpus) as pool:
        for settings in settings_lst:
            print(f"Done {done}/{len(settings_lst)}")
            err = []
            sim_time = []
            for n in range(0, len(amplitudes), cpus):
                print(n)
                results = pool.starmap(
                    perturb_and_run,
                    [(settings, A, N) for A in amplitudes[n:n+cpus]]
                )
                
                for result in results:
                    err.append(result["errnorm"])
                    sim_time.append(result["sim_time"])

            
            settings["err"] = err
            settings["sim_time"] = sim_time
            settings["amplitudes"] = amplitudes.tolist()

            done += 1

    
    with open(out_file, "w", newline="") as Ofile:
        json.dump(settings_lst, Ofile, indent=4)


