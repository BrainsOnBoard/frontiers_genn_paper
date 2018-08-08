import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import plot_settings
import utils

data = [("Jetson TX2", 33426.3, 12335.7, 19939.5),
        ("GeForce 1050ti", 15114.5, 1904.46, 2102.91),
        ("Tesla K40m", 4270.76, 1373.32, 1223.66),
        ("Tesla V100", 2218.93, 325.415, 419.117),
        ("HPC\n(fastest)", 3030.0, 0.0, 0.0),
        ("SpiNNaker", 20000, 0.0, 0.0)]

columns = zip(*data)
device = np.asarray(columns[0],  dtype=str)

total_sim_time = np.asarray(columns[1],  dtype=float) / 1000.0
neuron_sim_time = np.asarray(columns[2],  dtype=float) / 1000.0
synapse_sim_time = np.asarray(columns[3],  dtype=float) / 1000.0

overhead = total_sim_time - neuron_sim_time - synapse_sim_time

fig, axis = plt.subplots(figsize=(plot_settings.double_column_width, 90.0 * plot_settings.mm_to_inches),
                         frameon=False)

# Correctly place bars
bar_width = 0.8
bar_pad = 0.75
bar_x = np.arange(0.0, len(device) * (bar_width + bar_pad), bar_width + bar_pad)

offset = np.zeros(len(bar_x) - 2)

# Plot stacked, GPU bars
axis.bar(bar_x[:-2], neuron_sim_time[:-2], bar_width, label="Neuron simulation")
offset += neuron_sim_time[:-2]
axis.bar(bar_x[:-2], synapse_sim_time[:-2], bar_width, offset, label="Synapse simulation")
offset += synapse_sim_time[:-2]
axis.bar(bar_x[:-2], overhead[:-2], bar_width, offset, label="Overhead")
offset += overhead[:-2]

# Plot individual other bars
axis.bar(bar_x[-2:], total_sim_time[-2:], bar_width, 0.0)

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


# Set tight layour - tweaking bottom to fit in axis
# text and right to fit in right break marker
fig.tight_layout(pad=0, rect=(0.0, 0.0, 1.0, 0.96))
fig.savefig("../figures/microcircuit_performance.eps")
plt.show()