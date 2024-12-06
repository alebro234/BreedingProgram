# Plots results from the perturbative analysis 
# USAGE
# python plot_ptb.py -i out_ptb.json


import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",type=str,required=True)

    args = parser.parse_args()
    in_file = args.i

    with open(in_file, "r") as Ifile:
        settings_lst = json.load(Ifile)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for settings in settings_lst:
        label = (
            f"Cross: {settings['cross']}, "
            f"Mut: {settings['mut']}, "
            f"Pop: {settings['pop_size']}, "
            f"Genes: {settings['n_genes']}"
        )
        
        ax1.plot(
            settings["amplitudes"],
            [np.linalg.norm(e) for e in settings["err"]],
            label=label,
            linewidth=1.5  # Increase line width
        )
        ax1.set_xlabel("A %", fontsize=14)  # Larger x-axis label font size
        ax1.set_ylabel("Error Norm %", fontsize=14)  # Larger y-axis label font size
        ax1.tick_params(axis='both', which='major', labelsize=14)  # Larger axis ticks

        ax2.plot(
            settings["amplitudes"],
            settings["sim_time"],
            label=label,
            linewidth=1.5  # Increase line width
        )
        ax2.set_xlabel("A %", fontsize=14)  # Larger x-axis label font size
        ax2.set_ylabel("Simulation Time (s)", fontsize=14)  # Larger y-axis label font size
        ax2.tick_params(axis='both', which='major', labelsize=14)  # Larger axis ticks

# Place the legend in the middle of the figure
    handles, labels = ax1.get_legend_handles_labels()  # Collect all legend handles and labels
    fig.legend(
        handles, labels,
        loc="upper center",  # Place at the top center
        bbox_to_anchor=(0.5, 1),  # Fine-tune placement (centered above the figure)
        ncol=2,  # Number of columns for the legend entries
        fontsize=12,  # Legend font size
        title="Simulation Settings",
        title_fontsize=14  # Legend title font size
    )

# Create plots directory and save figure
    # os.makedirs("plots", exist_ok=True)
    # os.system("rm -f plots/" + out_file) 
    # plt.savefig(out_file)
    plt.show()

