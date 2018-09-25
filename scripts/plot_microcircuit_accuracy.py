import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import numpy as np
import plot_settings
import re
import utils

from os import path

from scipy.stats import entropy, gaussian_kde, iqr

from elephant.conversion import BinnedSpikeTrain
from elephant.statistics import isi, cv
from elephant.spike_train_correlation import corrcoef

from pandas import read_csv

from neo import SpikeTrain

from quantities import s, ms

N_full = {
  '23': {'E': 20683, 'I': 5834},
  '4' : {'E': 21915, 'I': 5479},
  '5' : {'E': 4850, 'I': 1065},
  '6' : {'E': 14395, 'I': 2948}
}

N_scaling = 1.0
duration = 9.0
raster_plot_step = 20
raster_plot_start_ms = 1000.0
raster_plot_end_ms = 2000.0

def load_spikes(filename):
    # Parse filename and use to get population name and size
    match = re.match("([0-9]+)([EI])\.csv", filename)
    name = match.group(1) + match.group(2)
    num = int(N_full[match.group(1)][match.group(2)] * N_scaling)

    # Load spikes
    spike_path = path.join("potjans_spikes", filename)
    spikes = read_csv(spike_path, header=None, skiprows=1, delimiter=",",
                      names=["time", "id"], dtype={"time":float, "id":int})

    # Convert CSV columns to numpy
    spike_times = spikes["time"].values
    spike_neuron_id = spikes["id"].values

    post_transient = (spike_times > 1000.0)
    spike_times = spike_times[post_transient]
    spike_neuron_id = spike_neuron_id[post_transient]

    # Load NEST spikes
    # **NOTE** retrospectively using NEO for all spike io would be better
    nest_spike_path = path.join("potjans_spikes", "nest", "spikes_L" + name + ".dat")
    nest_spikes = read_csv(nest_spike_path, header=None, comment="#", delimiter="\t",
                           names=["time", "id"], dtype={"time":float, "id":int})

    # Convert CSV columns to numpy
    nest_spike_times = nest_spikes["time"].values
    nest_spike_neuron_id = nest_spikes["id"].values
    nest_post_transient = (nest_spike_times > 1000.0)
    nest_spike_times = nest_spike_times[nest_post_transient]
    nest_spike_neuron_id = nest_spike_neuron_id[nest_post_transient]

    return spike_times, spike_neuron_id, name, num, nest_spike_times, nest_spike_neuron_id

def calc_histogram(data, smoothing, bin_x=None):
    if bin_x is None:
        # Calculate bin-size using Freedman-Diaconis rule
        bin_size = (2.0 * iqr(data)) / (float(len(data)) ** (1.0 / 3.0))

        # Get range of data
        min_y = np.amin(data)
        max_y = np.amax(data)

        # Calculate number of bins, rounding up to get right edge
        num_bins = np.ceil((max_y - min_y) / bin_size)

        # Create range of bin x coordinates
        bin_x = np.arange(min_y, min_y + (num_bins * bin_size), bin_size)

    # Create kernel density estimator of data
    data_kde = gaussian_kde(data, smoothing)

    # Use to generate smoothed histogram
    hist_smooth = data_kde.evaluate(bin_x)

    # Normalise histogram and return
    hist_normalised = hist_smooth / np.sum(hist_smooth) / (bin_x[1] - bin_x[0])
    return bin_x, hist_normalised

def calc_rate_hist(spike_times, spike_ids, num, duration, bin_x=None):
     # Calculate histogram of spike IDs to get each neuron's firing rate
    rate, _ = np.histogram(spike_ids, bins=range(num + 1))
    assert len(rate) == num
    rate = np.divide(rate, duration, dtype=float)
    
    return calc_histogram(rate, 0.3, bin_x)

def calc_cv_isi_hist(spike_times, spike_ids, num, duration, bin_x=None):
    # Loop through neurons
    cv_isi = []
    for n in range(num):
        # Get mask of spikes from this neuron and use to extract their times
        mask = (spike_ids == n)
        neuron_spike_times = spike_times[mask]
        
        # If this neuron spiked more than once i.e. it is possible to calculate ISI!
        if len(neuron_spike_times) > 1:
            cv_isi.append(cv(isi(neuron_spike_times)))

    return calc_histogram(cv_isi, 0.04, bin_x)

def calc_corellation(spike_times, spike_ids, num, duration, bin_x=None):
    # Create randomly shuffled indices
    neuron_indices = np.arange(num)
    np.random.shuffle(neuron_indices)

    # Loop through indices
    spike_trains = []
    for n in neuron_indices:
        # Extract spike times
        neuron_spike_times = spike_times[spike_ids == n]

        # If there are any spikes
        if len(neuron_spike_times) > 0:
            # Add neo SpikeTrain object
            spike_trains.append(SpikeTrain(neuron_spike_times * ms, t_start=1*s, t_stop=10*s))

            # If we have found our 200 spike trains, stop
            if len(spike_trains) == 200:
                break

    # Check that 200 spike trains containing spikes could be found
    assert len(spike_trains) == 200

    # Bin spikes using bins corresponding to 2ms refractory period
    binned_spike_trains = BinnedSpikeTrain(spike_trains, binsize=2.0 * ms)

    # Calculate correlation matrix
    correlation = corrcoef(binned_spike_trains)

    # Take lower triangle of matrix (minus diagonal)
    correlation_non_disjoint = correlation[np.tril_indices_from(correlation, k=-1)]

    # Calculate histogram
    return calc_histogram(correlation_non_disjoint, 0.002, bin_x)

pop_spikes = [load_spikes("6E.csv"),
              load_spikes("6I.csv"),
              load_spikes("5E.csv"),
              load_spikes("5I.csv"),
              load_spikes("4E.csv"),
              load_spikes("4I.csv"),
              load_spikes("23E.csv"),
              load_spikes("23I.csv")]

# Create plot
fig = plt.figure(figsize=(plot_settings.double_column_width, 90.0 * plot_settings.mm_to_inches),
                 frameon=False)

# Create outer gridspec dividing plot area into 4
gsp = gs.GridSpec(1, 4)

# Create sub-gridspecs for each panel of histograms with no spacing between axes
gs_rate_axes = gs.GridSpecFromSubplotSpec(4, 2, subplot_spec=gsp[1], wspace=0.0, hspace=0.0)
gs_cv_isi_axes = gs.GridSpecFromSubplotSpec(4, 2, subplot_spec=gsp[2], wspace=0.0, hspace=0.0)
gs_corr_axes = gs.GridSpecFromSubplotSpec(4, 2, subplot_spec=gsp[3], wspace=0.0, hspace=0.0)

# Add raster plot axis to figure
raster_axis = plt.Subplot(fig, gsp[0])
fig.add_subplot(raster_axis)
utils.remove_axis_junk(raster_axis)

# Get offsets of populations
neuron_id_offset = np.cumsum([0] + [n for _, _, _, n, _, _ in pop_spikes])

# Axis at start of each row and column to share x and y axes with
pop_rate_axis_col_sharex = [None] * 2
pop_rate_axis_row_sharey = [None] * 4
pop_cv_isi_axis_col_sharex = [None] * 2
pop_cv_isi_axis_row_sharey = [None] * 4
pop_corr_axis_col_sharex = [None] * 2
pop_corr_axis_row_sharey = [None] * 4

# Get palette
pal = sns.color_palette()

# Loop through populations
genn_actor = None
nest_actor = None
rate_kl = []
isi_kl = []
corr_kl = []
for i, (spike_times, spike_ids, name, num, nest_spike_times, nest_spike_ids) in enumerate(pop_spikes):
    col = i % 2
    row = i / 2

    # Plot the spikes from every raster_plot_step neurons within time range
    plot_mask = ((spike_ids % raster_plot_step) == 0) & (spike_times > raster_plot_start_ms) & (spike_times <= raster_plot_end_ms)
    raster_axis.scatter(spike_times[plot_mask], spike_ids[plot_mask] + neuron_id_offset[i], s=1, edgecolors="none")

    # Calculate statistics (using precise NEST stats to determine bins)
    rate_bin_x, nest_rate_hist = calc_rate_hist(nest_spike_times, nest_spike_ids, num, duration)
    _, rate_hist = calc_rate_hist(spike_times, spike_ids, num, duration, bin_x=rate_bin_x)
    isi_bin_x, nest_isi_hist = calc_cv_isi_hist(nest_spike_times, nest_spike_ids, num, duration)
    _, isi_hist = calc_cv_isi_hist(spike_times, spike_ids, num, duration, bin_x=isi_bin_x)
    corr_bin_x, nest_corr_hist = calc_corellation(nest_spike_times, nest_spike_ids, num, duration)
    _, corr_hist = calc_corellation(spike_times, spike_ids, num, duration, bin_x=corr_bin_x)

    # Create a mask to select bins where the GeNN simulation has non-negligible values
    rate_bin_mask = (rate_hist > 1.0E-15)
    isi_bin_mask = (isi_hist > 1.0E-15)
    corr_bin_mask = (corr_hist > 1.0E-15)

    # Calculate KL divergence
    rate_kl.append(entropy(nest_rate_hist[rate_bin_mask], rate_hist[rate_bin_mask]))
    isi_kl.append(entropy(nest_isi_hist[isi_bin_mask], isi_hist[isi_bin_mask]))
    corr_kl.append(entropy(nest_corr_hist[corr_bin_mask], corr_hist[corr_bin_mask]))

    assert np.isfinite(rate_kl[-1])
    assert np.isfinite(isi_kl[-1])
    assert np.isfinite(corr_kl[-1])

    # Plot rate histogram
    pop_rate_axis = plt.Subplot(fig, gs_rate_axes[3 - row, col],
                                sharex=pop_rate_axis_col_sharex[col],
                                sharey=pop_rate_axis_row_sharey[row])
    fig.add_subplot(pop_rate_axis)
    pop_rate_axis.text(1.0, 0.95, name, ha="right", va="top", transform=pop_rate_axis.transAxes)
    nest_actor =pop_rate_axis.plot(rate_bin_x, nest_rate_hist, color=pal[0])[0]
    genn_actor = pop_rate_axis.plot(rate_bin_x, rate_hist, linestyle=":", dash_capstyle="round", color=pal[2])[0]
    
    # Plot rate histogram
    pop_cv_isi_axis = plt.Subplot(fig, gs_cv_isi_axes[3 - row, col],
                                  sharex=pop_cv_isi_axis_col_sharex[col],
                                  sharey=pop_cv_isi_axis_row_sharey[row])
    fig.add_subplot(pop_cv_isi_axis)
    pop_cv_isi_axis.text(1.0, 0.95, name, ha="right", va="top", transform=pop_cv_isi_axis.transAxes)
    pop_cv_isi_axis.plot(isi_bin_x, nest_isi_hist, color=pal[0])
    pop_cv_isi_axis.plot(isi_bin_x, isi_hist, linestyle=":", dash_capstyle="round", color=pal[2])

    # Plot correlation histogram
    pop_corr_axis = plt.Subplot(fig, gs_corr_axes[3 - row, col],
                                sharex=pop_corr_axis_col_sharex[col],
                                sharey=pop_corr_axis_row_sharey[row])
    fig.add_subplot(pop_corr_axis)
    pop_corr_axis.text(1.0, 0.95, name, ha="right", va="top", transform=pop_corr_axis.transAxes)
    pop_corr_axis.plot(corr_bin_x, nest_corr_hist, color=pal[0])
    pop_corr_axis.plot(corr_bin_x, corr_hist, linestyle=":", dash_capstyle="round", color=pal[2])

    # Remove axis junk
    utils.remove_axis_junk(pop_rate_axis)
    utils.remove_axis_junk(pop_cv_isi_axis)
    utils.remove_axis_junk(pop_corr_axis)

    # If this is the first (leftmost) column
    if col == 0:
        # Cache axes so that their Y axis can be shared with others in same row
        pop_rate_axis_row_sharey[row] = pop_rate_axis
        pop_cv_isi_axis_row_sharey[row] = pop_cv_isi_axis
        pop_corr_axis_row_sharey[row] = pop_corr_axis

        # Set y axis labels
        pop_rate_axis.set_ylabel("p")
        pop_cv_isi_axis.set_ylabel("p")
        pop_corr_axis.set_ylabel("p")
    # Otherwise, hide y axes
    else:
        plt.setp(pop_rate_axis.get_yticklabels(), visible=False)
        plt.setp(pop_cv_isi_axis.get_yticklabels(), visible=False)
        plt.setp(pop_corr_axis.get_yticklabels(), visible=False)

    # If this is the first (bottommost) row
    if row == 0:
        # Cache axes so that their X axis can be shared with others in same column
        pop_rate_axis_col_sharex[col] = pop_rate_axis
        pop_cv_isi_axis_col_sharex[col] = pop_cv_isi_axis
        pop_corr_axis_col_sharex[col] = pop_corr_axis

        # Set x axis labels
        pop_rate_axis.set_xlabel("rate\n[spikes/s]")
        pop_cv_isi_axis.set_xlabel("CV ISI")
        pop_corr_axis.set_xlabel("corr.\ncoef.")
    # Otherwise, hide x axes
    else:
        plt.setp(pop_rate_axis.get_xticklabels(), visible=False)
        plt.setp(pop_cv_isi_axis.get_xticklabels(), visible=False)
        plt.setp(pop_corr_axis.get_xticklabels(), visible=False)

    # If this is the top left cell add subfigure title
    if row == 3 and col == 0:
        pop_rate_axis.set_title("B", loc="left")
        pop_cv_isi_axis.set_title("C", loc="left")
        pop_corr_axis.set_title("D", loc="left")

# Set axis y limits on axes in first column of each block of axes
# **NOTE** this needs to be done after sharing etc is setup
for a in pop_rate_axis_row_sharey:
    a.set_ylim((-0.1, 0.9))
for a in pop_cv_isi_axis_row_sharey:
    a.set_ylim((-0.5, 4.5))
for a in pop_corr_axis_row_sharey:
    a.set_ylim((-50.0, 650.0))

# Set axis x limits and ticks on axes in first row of each block of axes
# **NOTE** this needs to be done after sharing etc is setup
for a in pop_rate_axis_col_sharex:
    a.set_xlim((0.0, 15.0))
    a.set_xticks([0.0, 10.0])
for a in pop_cv_isi_axis_col_sharex:
    a.set_xlim((0.0, 1.6))
    a.set_xticks([0.0, 1.0])
for a in pop_corr_axis_col_sharex:
    a.set_xlim((-0.03, 0.05))
    a.set_xticks((0.0, 0.04))

raster_axis.set_xlabel("Time [ms]")

pop_names = [name for _, _, name, _, _,_ in pop_spikes]
# Calculate midpoints of each population in terms of neuron ids and position population ids here
pop_midpoints = neuron_id_offset[:-1] + ((neuron_id_offset[1:] - neuron_id_offset[:-1]) * 0.5)
raster_axis.set_yticks(pop_midpoints)
raster_axis.set_yticklabels(pop_names)
raster_axis.set_xlim((raster_plot_start_ms, raster_plot_end_ms))
raster_axis.set_ylim((0, neuron_id_offset[-1]))

raster_axis.set_title("A", loc="left")

fig.legend([genn_actor, nest_actor], ["GeNN", "NEST 'precise'"],
           ncol=2, loc="lower center")

fig.tight_layout(pad=0.0, rect=(0.0, 0.075, 1, 0.99))

# Save figure
utils.save_raster_figure(fig, "../figures/microcircuit_accuracy")

# Create second figure to show KL divergence
kl_fig, kl_axes = plt.subplots(3, figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches),
                               frameon=False)

# Position bars
kl_bar_width = 0.8
kl_bar_pad = 0.4
kl_bar_x = np.arange(0.0, len(rate_kl) * (kl_bar_width + kl_bar_pad), kl_bar_width + kl_bar_pad)

# Values for grid-based NEST and SpiNNaker eye-balled from paper in pixels with page loaded at 2000px wide
rate_kl_seeds =     np.asarray([21,     31,     82,     10,     24, 8,      27,     5], dtype=float)
rate_kl_nest =      np.asarray([32,     7,      132,    12,     11, 6,      28,     50], dtype=float)
rate_kl_spinnaker = np.asarray([63,     13,     68,     33,     21, 11,     29,     21], dtype=float)

isi_kl_seeds =      np.asarray([62,     6,      47,     18,     34, 8,      26,     4], dtype=float)
isi_kl_nest =       np.asarray([121,    10,     56,     16,     31, 10,     33,     6], dtype=float)
isi_kl_spinnaker =  np.asarray([83,     5,      135,    22,     28, 8,      17,     3], dtype=float)

corr_kl_seeds =     np.asarray([66,     92,     45,     44,     66, 82,     93,     60], dtype=float)
corr_kl_nest =      np.asarray([47,     75,     50,     102,    87, 65,     91,     55], dtype=float)
corr_kl_spinnaker = np.asarray([70,     77,     38,     43,     44, 103,    107,    91], dtype=float)

# Scale eye-balled values from pixels to actual units
# **YUCK** ISI is blotted with 2 different scales
rate_kl_nest *= (0.005 / 75.0)
rate_kl_spinnaker *= (0.005 / 75.0)
rate_kl_seeds *= (0.005 / 75.0)

isi_kl_nest[:2] *= (0.05 / 75.0)
isi_kl_spinnaker[:2] *= (0.05 / 75.0)
isi_kl_seeds[:2] *= (0.05 / 75.0)

isi_kl_nest[2:4] *= (0.1 / 75.0)
isi_kl_spinnaker[2:4] *= (0.1 / 75.0)
isi_kl_seeds[2:4] *= (0.1 / 75.0)

isi_kl_nest[4:] *= (0.05 / 75.0)
isi_kl_spinnaker[4:] *= (0.05 / 75.0)
isi_kl_seeds[4:] *= (0.05 / 75.0)

corr_kl_nest *= (0.1 / 75.0)
corr_kl_spinnaker *= (0.1 / 75.0)
corr_kl_seeds *= (0.1 / 75.0)

# Plot bars
seeds_actor = kl_axes[0].bar(kl_bar_x * 4, rate_kl_seeds, kl_bar_width)[0]
nest_actor = kl_axes[0].bar(kl_bar_x * 4 + kl_bar_width, rate_kl_nest, kl_bar_width)[0]
spinnaker_actor = kl_axes[0].bar(kl_bar_x * 4 + kl_bar_width * 2, rate_kl_spinnaker, kl_bar_width)[0]
genn_actor = kl_axes[0].bar(kl_bar_x * 4 + kl_bar_width * 3, rate_kl, kl_bar_width)[0]

kl_axes[1].bar(kl_bar_x * 4, isi_kl_seeds, kl_bar_width)
kl_axes[1].bar(kl_bar_x * 4 + kl_bar_width, isi_kl_nest, kl_bar_width)
kl_axes[1].bar(kl_bar_x * 4 + kl_bar_width * 2, isi_kl_spinnaker, kl_bar_width)
kl_axes[1].bar(kl_bar_x * 4 + kl_bar_width * 3, isi_kl, kl_bar_width)

kl_axes[2].bar(kl_bar_x * 4, corr_kl_seeds, kl_bar_width)
kl_axes[2].bar(kl_bar_x * 4 + kl_bar_width, corr_kl_nest, kl_bar_width)
kl_axes[2].bar(kl_bar_x * 4 + kl_bar_width * 2, corr_kl_spinnaker, kl_bar_width)
kl_axes[2].bar(kl_bar_x * 4 + kl_bar_width * 3, corr_kl, kl_bar_width)

# Set axis labels and titles
for axis, title in zip(kl_axes, ["A", "B", "C"]):
    axis.set_ylabel("$D_{KL}$")
    axis.set_title(title, loc="left")
    axis.set_xticks(kl_bar_x * 4)
    axis.set_xticklabels(pop_names, ha="center")

kl_fig.legend([seeds_actor, nest_actor, spinnaker_actor, genn_actor],
              ["NEST 'precise' different seed", "NEST 'grid-aligned'", "SpiNNaker", "GeNN"],
              ncol=2, loc="lower center")

kl_fig.tight_layout(pad=0, rect=(0, 0.15, 1, 1))
kl_fig.savefig("../figures/microcircuit_accuracy_kl.eps")

# Show plot
plt.show()

