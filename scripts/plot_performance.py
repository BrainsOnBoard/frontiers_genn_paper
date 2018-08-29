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
    group = None if group_size is None else np.asarray(columns[1],  dtype=str)
    time_col_start = 1 if group_size is None else 2

    # Read times into numpy arrays
    num_time_columns = len(columns) - time_col_start
    times = np.empty((num_time_columns, len(device)), dtype=float)
    for i, col in enumerate(columns[time_col_start:]):
        times[i,:] = col
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


    # If there are no groups space bars evenly
    group_x = []
    if group_size is None:
        bar_pad = 0.4
        bar_x = np.arange(0.0, len(device) * (bar_width + bar_pad), bar_width + bar_pad)
    # Otherwise
    else:
        bar_pad = 0.1
        group_pad = 0.75
        start = 0.0
        bar_x = np.empty(len(device))

        # Calculate bar positions of grouped GPU bars
        for d in range(0, len(device) - num_ref, group_size):
            end = start + ((bar_width + bar_pad) * group_size)
            bar_x[d:d + group_size] = np.arange(start, end, bar_width + bar_pad)

            group_x.append(start + ((end - bar_width - start) * 0.5))

            # Update start for next group
            start = end + group_pad

        # Calculate bar positions of other bars
        for d in range(len(device) - num_ref, len(device)):
            bar_x[d] = start
            group_x.append(start)# + (bar_width * 0.5))

            # Update start for next group
            start += (bar_width + group_pad)

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

    # Default tight layout rectangle
    tight_layout_rect = [0.0, 0.0, 1.0, 1.0]

    # If there are no groups, use device names as x tick labels
    if group_size is None:
        axis.set_xticklabels(device, rotation="vertical", ha="center", multialignment="right")
    # Otherwise
    else:
        # Use group names as x tick labels
        axis.set_xticklabels(group, rotation="vertical", ha="center", multialignment="right")

        # Get name of device associated with each group and use these as x-ticks
        unique_device = np.hstack((device[0:-num_ref:group_size], device[-num_ref:]))

        # Add extra text labelling the device associated with each device
        for x, s in zip(group_x, unique_device):
            # **YUCK** because of potential log scale, using data coordinates here is tricky SO
            # First convert position of group along x-axis into display coordinates
            x_disp = axis.transData.transform_point((x, 0))

            # Then transform THAT into axis coordinates
            x_axis = axis.transAxes.inverted().transform_point(x_disp)

            # Draw text offset from x-axis in axes coordinates
            axis.text(x_axis[0], -0.25, s, rotation="vertical", ha="center", va="top", multialignment="right",
                      clip_on=False, transform=axis.transAxes)

        # Tweak tight layout rect to fit in extra text
        tight_layout_rect[1] += 0.25

    # Set log scale if required
    if log:
        axis.set_yscale("log", nonposy="clip")

    # If legend text is specified
    if legend_text is not None:
        # Add legend
        fig.legend(legend_actors, legend_text, ncol=2, loc="lower center")

        # Tweak bottom of tight layout rect to fit in legend
        tight_layout_rect[1] += 0.15

    # Set tight layout and save
    fig.tight_layout(pad=0, rect=tight_layout_rect)
    fig.savefig(filename)

# Total simulation time, neuron simulation, synapse simulation
microcircuit_data = [("Jetson TX2", 99570.4, 155284, 258350),
                     ("GeForce 1050ti", 20192.6, 21310.1, 137592),
                     ("Tesla K40c", 13636.2, 12431.8, 41911.5),
                     ("Tesla V100", 3215.88, 3927.9, 21645.4),
                     ("HPC\n(fastest)", 0.0, 0.0, 24296.0),
                     ("SpiNNaker", 0.0, 0.0, 200000)]

microcircuit_init_data = [("Jetson\nTX2", "Device", 753.284 + 950.965 + 1683.32),
                          ("Jetson\nTX2", "Host", 125.569 + 14.438 + 541196 + 85984.6),
                          ("GeForce\n1050ti", "Device", 347.681 + 499.292 + 561.601),
                          ("GeForce\n1050ti", "Host", 362.013 + 7.14622 + 19110 + 49768.2),
                          ("Tesla\nK40c", "Device", 204.258 + 361.698 + 392.913),
                          ("Tesla\nK40c", "Host", 18522.8),
                          ("Tesla\nV100", "Device", 58.6588 + 142.279 + 445.239),
                          ("Tesla\nV100", "Host", 16182.2),
                          ("HPC\n(fastest)", "", 2000.0),
                          ("SpiNNaker", "", 10.0 * 60.0 * 60.0 * 1000.0)]

# Total simulation time, neuron simulation, synapse simulation, postsynaptic learning
stdp_data = [("Tesla K40m\nBitmask", 435387, 296357, 3925070, 4736610),
             ("Tesla V100\nBitmask", 100144, 82951.6, 307273, 564826),
             ("Tesla V100\nRagged", 99346.3, 85975.4, 307433, 567267)]

plot(microcircuit_init_data, "../figures/microcircuit_init_performance.eps", 2, False,
     None, None, 2, True)

plot(microcircuit_data, "../figures/microcircuit_performance.eps", 2, True,
     ["Neuron simulation", "Synapse\nsimulation", "Overhead"], 10.0)

plot(stdp_data, "../figures/stdp_performance.eps", 0, True,
     ["Neuron simulation", "Synapse simulation", "Postsynaptic learning", "Overhead"], 200.0)

plt.show()