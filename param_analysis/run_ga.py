import sys
import json
import argparse
sys.path.append("/home/ale234/github/BreedingProgram")
from BreedingProgram import run_breeder  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    input_file = args.input

    with open(input_file, "r") as f:
        settings_lst = json.load(f)

    for settings in settings_lst:
        best = run_breeder(settings)

        print(f"Completed in {best.id} generations")
        best.display_genome(decoded=True)

        
