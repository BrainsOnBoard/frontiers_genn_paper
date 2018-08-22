import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import plot_settings
import utils

def plot(data, filename, num_ref):
    columns = zip(*data)
    device = np.asarray(columns[0],  dtype=str)

    total_sim_time = np.asarray(columns[1],  dtype=float) / 1000.0
    neuron_sim_time = np.asarray(columns[2],  dtype=float) / 1000.0
    synapse_sim_time = np.asarray(columns[3],  dtype=float) / 1000.0

    overhead = total_sim_time - neuron_sim_time - synapse_sim_time

    fig, axis = plt.subplots(figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches),
                            frameon=False)

    # Correctly place bars
    bar_width = 0.8
    bar_pad = 0.4
    bar_x = np.arange(0.0, len(device) * (bar_width + bar_pad), bar_width + bar_pad)

    offset = np.zeros(len(bar_x) - num_ref)

    # Plot stacked, GPU bars
    neuron_sim_actor = axis.bar(bar_x[:-num_ref], neuron_sim_time[:-num_ref], bar_width)[0]
    offset += neuron_sim_time[:-num_ref]
    synapse_sim_actor = axis.bar(bar_x[:-num_ref], synapse_sim_time[:-num_ref], bar_width, offset)[0]
    offset += synapse_sim_time[:-num_ref]
    overhead_actor = axis.bar(bar_x[:-num_ref], overhead[:-num_ref], bar_width, offset)
    offset += overhead[:-num_ref]

    # Plot individual other bars
    axis.bar(bar_x[-num_ref:], total_sim_time[-num_ref:], bar_width, 0.0)

    # Add real-timeness annoation
    #for t, x in zip(total_sim_time, bar_x):
    #    axis.text(x, t,
    #              "%.2f$\\times$\nreal-time" % (1.0 / t),
    #              ha="center", va="bottom", )

    axis.set_ylabel("Time [s]")

    # Add legend
    axis.legend(loc="upper right", ncol=3)

    # Add realtime line
    axis.axhline(1.0, color="black", linestyle="--")

    # Remove vertical grid
    axis.xaxis.grid(False)

    # Add x ticks labelling delay type
    axis.set_xticks(bar_x)
    axis.set_xticklabels(device, rotation="vertical", ha="center", multialignment="right")

    fig.legend([neuron_sim_actor, synapse_sim_actor, overhead_actor],
            ["Neuron\nsimulation", "Synapse\nsimulation", "Overhead"],
            ncol=3, loc="lower center")

    # Set tight layour - tweaking bottom to fit in axis
    # text and right to fit in right break marker
    fig.tight_layout(pad=0, rect=(0.0, 0.09, 1.0, 0.96))
    fig.savefig(filename)

microcircuit_data = [("Jetson TX2", 29335, 13165.9, 15676.7),
                     ("GeForce 1050ti", 14102.4, 2159.49, 2106.02),
                     ("Tesla K40m", 4270.76, 1373.32, 1223.66),
                     ("Tesla V100", 2244.04, 370.681, 392.14),
                     ("HPC\n(fastest)", 3030.0, 0.0, 0.0),
                     ("SpiNNaker", 20000, 0.0, 0.0)]

stdp_data = [("Tesla K40m", 4736610, 435387, 296357),
             ("Tesla V100\nBitmask", 564826, 100144, 82951.6),
             ("Tesla V100\nRagged", 567267, 99346.3, 85975.4),
             ("HPC", 60.0 * 60.0 * 60.0 * 1000.0 / 5.0, 0.0, 0.0)]

plot(microcircuit_data, "../figures/microcircuit_performance.eps", 2)
plot(stdp_data, "../figures/stdp_performance.eps", 1)
plt.show()