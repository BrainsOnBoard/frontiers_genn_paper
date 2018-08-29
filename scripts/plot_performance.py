import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import plot_settings
import utils

def plot(data, filename, num_ref, calc_overhead, legend_text, real_time_s=None, group_size=None, log=False):
    columns = zip(*data)
    device = np.asarray(columns[0],  dtype=str)

    time_col_start = 1 if group_size is None else 2

    # Read times into numpy arrays
    num_time_columns = len(columns) - time_col_start
    times = np.empty((num_time_columns, len(device)), dtype=float)
    for i, col in enumerate(columns[time_col_start:]):
        times[i,:] = col
    print times
    # Convert ms to s
    times /= 1000.0


    # If overheads are being calculated, subtract all preceeding rows from last row
    if calc_overhead:
        for i in range(times.shape[0] - 1):
            times[-1,:-num_ref] -= times[i,:-num_ref]

    fig, axis = plt.subplots(figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches),
                             frameon=False)

    # Correctly place bars
    bar_width = 0.8
    bar_pad = 0.4
    bar_x = np.arange(0.0, len(device) * (bar_width + bar_pad), bar_width + bar_pad)

    offset = np.zeros(len(bar_x) - num_ref)

    # Plot stacked, GPU bars
    gpu_bar_x_slice = np.s_[:] if num_ref == 0 else np.s_[:-num_ref]

    legend_actors = []
    for i in range(times.shape[0]):
        legend_actors.append(axis.bar(bar_x[gpu_bar_x_slice], times[i,gpu_bar_x_slice], bar_width, offset)[0])
        offset += times[i,gpu_bar_x_slice]

    # Plot individual other bars
    if num_ref > 0:
        print times[-1,-num_ref:]
        axis.bar(bar_x[-num_ref:], times[-1,-num_ref:], bar_width, 0.0)

    axis.set_ylabel("Time [s]")

    # Add legend
    axis.legend(loc="upper right", ncol=3)

    # Add realtime line
    if real_time_s is not None:
        axis.axhline(real_time_s, color="black", linestyle="--")

    # Remove vertical grid
    axis.xaxis.grid(False)

    # Add x ticks
    axis.set_xticks(bar_x)
    axis.set_xticklabels(device, rotation="vertical", ha="center", multialignment="right")

    if log:
        axis.set_yscale("log", nonposy="clip")

    # Add legend
    fig.legend(legend_actors, legend_text, ncol=2, loc="lower center")

    # Set tight layour - tweaking bottom to fit in axis
    # text and right to fit in right break marker
    fig.tight_layout(pad=0, rect=(0.0, 0.15, 1.0, 0.96))
    fig.savefig(filename)

# Total simulation time, neuron simulation, synapse simulation
microcircuit_data = [("Jetson TX2", 99570.4, 155284, 258350),
                     ("GeForce 1050ti", 20192.6, 21310.1, 137592),
                     ("Tesla K40c", 13636.2, 12431.8, 41911.5),
                     ("Tesla V100", 3215.88, 3927.9, 21645.4),
                     ("HPC\n(fastest)", 0.0, 0.0, 24296.0),
                     ("SpiNNaker", 0.0, 0.0, 200000)]

microcircuit_init_data = [("Jetson TX2", "Device\ninit", 753.284 + 950.965, 1683.32),
                          ("Jetson TX2", "Host\ninit", 125.569, 14.438 + 541196 + 85984.6),
                          ("GeForce 1050ti", "Device\ninit", 347.681 + 499.292, 561.601),
                          ("GeForce 1050ti", "Host\ninit", 362.013, 7.14622 + 19110 + 49768.2),
                          ("Tesla K40c", "Device\ninit", 204.258 + 361.698, 392.913),
                          ("Tesla K40c", "Host\ninit", 0.0, 18522.8),
                          ("Tesla V100", "Device\ninit", 58.6588 + 142.279, 445.239),
                          ("Tesla V100", "Host\ninit", 0.0, 16182.2),
                          ("HPC\n(fastest)", "", 0.0, 2000.0),
                          ("SpiNNaker", "", 0.0, 10.0 * 60.0 * 60.0 * 1000.0)]

# Total simulation time, neuron simulation, synapse simulation, postsynaptic learning
stdp_data = [("Tesla K40m\nBitmask", 435387, 296357, 3925070, 4736610),
             ("Tesla V100\nBitmask", 100144, 82951.6, 307273, 564826),
             ("Tesla V100\nRagged", 99346.3, 85975.4, 307433, 567267)]

plot(microcircuit_init_data, "../figures/microcircuit_init_performance.eps", 2, False,
     ["Device initialisation", "Host initialisation"], None, 2, True)

plot(microcircuit_data, "../figures/microcircuit_performance.eps", 2, True,
     ["Neuron simulation", "Synapse\nsimulation", "Overhead"], 10.0)

plot(stdp_data, "../figures/stdp_performance.eps", 0, True,
     ["Neuron simulation", "Synapse simulation", "Postsynaptic learning", "Overhead"], 200.0)

plt.show()