import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=False)

args = parser.parse_args()
input_file = args.input

if input_file is None:
    print("No input file provided")
