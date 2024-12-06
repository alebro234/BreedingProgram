#! usr/bin/bash
clear
py perturb_test.py --cpus 8 -i ../optimized.json -o out_ptb_entr.json
py perturb_test.py --cpus 8 -i ../optimized.json -o out_ptb_entr.json
py plot_ptb -i out_ptb_entr.json
py plot_ptb -i out_ptb_tourn.json
