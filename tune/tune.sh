#! usr/bin/bash
clear
py self_tune.py --cpus 8 -i input_entropy.json -o ../optimized.json
py self_tune.py --cpus 8 -i input_tournament.json -o ../optimized.json
