import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import plot_settings
import utils

def plot(data, filename, num_ref, calc_overhead, legend_text, real_time_s=None, group_size=None, log=False):
    columns = list(zip(*data))
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
            if num_ref == 0:
                times[-1,:] -= times[i,:]
            else:
                times[-1,:-num_ref] -= times[i,:-num_ref]

    fig, axis = plt.subplots(figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches),
                             frameon=False)

    # Correctly place bars
    bar_width = 0.8

    # If there are no groups space bars evenly
    group_x = []
    if group_size is None:
        bar_pad = 0.4
        bar_x = np.asarray([float(d) * (bar_width + bar_pad)
                            for d in range(len(device))])
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

    assert len(bar_x) == len(device)
    offset = np.zeros(len(bar_x) - num_ref)

    # Plot stacked, GPU bars
    gpu_bar_x_slice = np.s_[:] if num_ref == 0 else np.s_[:-num_ref]

    pal = sns.color_palette("deep")
    pal = [pal[0], pal[2], pal[3]] + pal[4:]
    legend_actors = []
    for i in range(times.shape[0]):
        # Build colour vector - colouring bars based on group and stack height
        colour = None
        num_bars = len(device) - num_ref
        if group_size is not None and num_time_columns == 1:
            colour = [pal[(i * group_size) + j] for j in range(group_size)] * num_bars
        else:
            colour = [pal[i]] * num_bars

        bars = axis.bar(bar_x[gpu_bar_x_slice], times[i,gpu_bar_x_slice], bar_width, offset, color=colour, linewidth=0)

        if group_size is not None and num_time_columns == 1:
            legend_actors.extend(b for b in bars[:group_size])
        else:
            legend_actors.append(bars[0])

        offset += times[i,gpu_bar_x_slice]

    # Plot individual other bars
    if num_ref > 0:
        colour = pal[times.shape[0]] if group_size is None else pal[times.shape[0] * group_size]
        axis.bar(bar_x[-num_ref:], times[-1,-num_ref:], bar_width, 0.0, color=colour, linewidth=0)

    axis.set_ylabel("Time [s]")

    # Add realtime line
    if real_time_s is not None:
        axis.axhline(real_time_s, color="black", linestyle="--")

    # Remove vertical grid
    axis.xaxis.grid(False)

    # Default tight layout rectangle
    tight_layout_rect = [0.0, 0.0, 1.0, 1.0]

    # If there are no groups, use device names as x tick labels
    if group_size is None:
        axis.set_xticks(bar_x)
        axis.set_xticklabels(device, rotation="vertical", ha="center", multialignment="right")
    # Otherwise
    else:
        # Use device names as x tick labels
        if num_ref > 0:
            unique_device = np.hstack((device[0:-num_ref:group_size], device[-num_ref:]))
        else:
            unique_device = device[::group_size]

        # If the legend is already being used for other purposes
        if legend_text is not None:
            # Use x-ticks to display which group each bar is in
            axis.set_xticks(bar_x[:-num_ref])
            axis.set_xticklabels(np.tile(group[:group_size], len(group_x) - num_ref),
                                 rotation="vertical", ha="center", multialignment="right", size="x-small")

            # Add additional axis text to display which device each group is in
            for x, d in zip(group_x, unique_device):
                axis.text(x, -11, d, horizontalalignment="center", va="top", size="medium")

            # Add a bit of extra space to bounding rectangle
            tight_layout_rect[1] += 0.12
        else:
            # Place x tick at centre of each group and label with device names
            axis.set_xticks(group_x)
            axis.set_xticklabels(unique_device, rotation="vertical", ha="center", multialignment="right")

    # Set log scale if required
    if log:
        axis.set_yscale("log", nonposy="clip")

    # If legend text is specified
    if legend_text is not None:
        # Add legend
        fig.legend(legend_actors, legend_text,
                   ncol=2,
                   loc="lower center")

        # Tweak bottom of tight layout rect to fit in legend
        if not plot_settings.presentation:
            tight_layout_rect[1] += 0.15
    elif group is not None:
        # Add legend
        fig.legend(legend_actors[:group_size], group[:group_size],
                   ncol=group_size if plot_settings.presentation else 2,
                   loc="lower center")

        # Tweak bottom of tight layout rect to fit in legend
        if not plot_settings.presentation:
            tight_layout_rect[1] += 0.1

    # Set tight layout and save
    fig.tight_layout(pad=0, rect=tight_layout_rect)
    if not plot_settings.presentation:
        fig.savefig(filename)

# neuron simulation, synapse simulation, Total simulation time,
microcircuit_data = [("Jetson\nTX2", "GeNN 3.2.0", 10023, 15632, 26076),
                     ("Jetson\nTX2", "Latest", 8176, 12259, 21280),
                     ("GeForce\n1650", "GeNN 3.2.0", 0, 0, 0),
                     ("GeForce\n1650", "Latest", 1103, 1683, 3940),
                     ("Tesla\nK40c", "GeNN 3.2.0", 1363, 1247, 4206),
                     ("Tesla\nK40c", "Latest", 794, 1074, 3580),
                     ("Titan\nV", "GeNN 3.2.0", 413, 463, 1838),
                     ("Titan\nV", "Latest", 288, 380, 1656),
                     ("HPC\n(fastest)", "", 0.0, 0.0, 2429.60)]

microcircuit_init_data = [("Jetson TX2", "GPU initialisation", 753.284 + 950.965 + 1683.32),
                          ("Jetson TX2", "CPU initialisation", 125.569 + 14.438 + 541196 + 85984.6),
                          ("GeForce 1050ti", "GPU initialisation", 347.681 + 499.292 + 561.601),
                          ("GeForce 1050ti", "CPU initialisation", 362.013 + 7.14622 + 19110 + 49768.2),
                          ("Tesla K40c", "GPU initialisation", 204.258 + 361.698 + 392.913),
                          ("Tesla K40c", "CPU initialisation", 18522.8),
                          ("Tesla V100", "GPU initialisation", 58.6588 + 142.279 + 445.239),
                          ("Tesla V100", "CPU initialisation", 16182.2),
                          ("HPC (fastest)", "", 2000.0),
                          ("SpiNNaker", "", 10.0 * 60.0 * 60.0 * 1000.0)]

microcircuit_build_data = [("Jetson TX2", "GeNN 3.2.0", 1000 * ((3 * 60) + 13)),
                           ("Jetson TX2", "Latest", 36000),
                           ("GeForce 1650", "GeNN 3.2.0", 0),
                           ("GeForce 1650", "Latest", 12000),
                           ("Tesla K40c", "GeNN 3.2.0", (60 + 23) * 1000),
                           ("Tesla K40c", "Latest", 23000),
                           ("Titan V", "GeNN 3.2.0", (60 + 6) * 1000),
                           ("Titan V", "Latest", 11800)]

# Total simulation time, neuron simulation, synapse simulation, postsynaptic learning
stdp_data = [("Tesla K40c\nBitmask", 529559, 754827, 9149110, 10530000),
             ("Tesla V100\nBitmask", 120379, 206731, 710839, 1118660),
             ("Tesla V100\nStandard", 120446, 210367, 715422, 1127640)]

plot(microcircuit_init_data, "../figures/microcircuit_init_performance.eps", 2, False,
     None, None, 2, True)

plot(microcircuit_build_data, "../figures/microcircuit_build_performance.eps", 0, False,
     None, None, 2)

plot(microcircuit_data, "../figures/microcircuit_performance.eps", 1, True,
     ["Neuron simulation", "Synapse simulation", "Overhead"], 1.0, 2)

plot(stdp_data, "../figures/stdp_performance.eps", 0, True,
     ["Neuron simulation", "Synapse simulation", "Postsynaptic learning", "Overhead"], 200.0)

plt.show()
