import csv
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import re
import plot_settings
import warnings

from os import path
from scipy.stats import gaussian_kde
from scipy.stats import iqr

from elephant.conversion import BinnedSpikeTrain
from elephant.statistics import isi, cv
from elephant.spike_train_correlation import corrcoef

from neo import SpikeTrain
from neo.io import PickleIO
from quantities import s, ms

N_full = {
  '23': {'E': 20683, 'I': 5834},
  '4' : {'E': 21915, 'I': 5479},
  '5' : {'E': 4850, 'I': 1065},
  '6' : {'E': 14395, 'I': 2948}
}

N_scaling = 1.0
duration = 9.0

def load_spikes(filename):
    # Parse filename and use to get population name and size
    match = re.match("([0-9]+)([EI])\.csv", filename)
    name = match.group(1) + match.group(2)
    num = int(N_full[match.group(1)][match.group(2)] * N_scaling)

    # Load spikes
    spike_path = path.join("potjans_spikes", filename)
    spikes = np.loadtxt(spike_path, skiprows=1, delimiter=",",
                        dtype={"names": ("time", "id", ), "formats": (float, int)})

    # Convert CSV columns to numpy
    spike_times = spikes["time"]
    spike_neuron_id = spikes["id"]

    post_transient = (spike_times > 1000.0)
    spike_times = spike_times[post_transient]
    spike_neuron_id = spike_neuron_id[post_transient]

    return spike_times, spike_neuron_id, name, num

def calc_histogram(data, smoothing):
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
    
    # Return
    return bin_x, hist_smooth

def calc_rate_hist(spike_times, spike_ids, num, duration):
     # Calculate histogram of spike IDs to get each neuron's firing rate
    rate, _ = np.histogram(spike_ids, bins=range(num + 1))
    assert len(rate) == num
    rate = np.divide(rate, duration, dtype=float)
    
    return calc_histogram(rate, 0.3)

def calc_cv_isi_hist(spike_times, spike_ids, num, duration):
    # Loop through neurons
    cv_isi = []
    for n in range(num):
        # Get mask of spikes from this neuron and use to extract their times
        mask = (spike_ids == n)
        neuron_spike_times = spike_times[mask]
        
        # If this neuron spiked more than once i.e. it is possible to calculate ISI!
        if len(neuron_spike_times) > 1:
            cv_isi.append(cv(isi(neuron_spike_times)))

    return calc_histogram(cv_isi, 0.04)

def calc_corellation(spike_times, spike_ids, num, duration):
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
    return calc_histogram(correlation_non_disjoint, 0.002)

pop_spikes = [load_spikes("6I.csv"),
              load_spikes("6E.csv"),
              load_spikes("5I.csv"),
              load_spikes("5E.csv"),
              load_spikes("4I.csv"),
              load_spikes("4E.csv"),
              load_spikes("23I.csv"),
              load_spikes("23E.csv")]

# Create plot
fig = plt.figure(figsize=(plot_settings.double_column_width, 90.0 * plot_settings.mm_to_inches),
                 frameon=False)
gsp = gs.GridSpec(4, 8)

# Add raster plot axis to figure
raster_axis = plt.Subplot(fig, gsp[:,:2])
fig.add_subplot(raster_axis)

start_id = 0
for i, (spike_times, spike_ids, name, num) in enumerate(pop_spikes):
    # Plot spikes
    raster_axis.scatter(spike_times, spike_ids + start_id, s=1, edgecolors="none")

    # Calculate statistics
    rate_bin_x, rate_hist = calc_rate_hist(spike_times, spike_ids, num, duration)
    isi_bin_x, isi_hist = calc_cv_isi_hist(spike_times, spike_ids, num, duration)
    corr_bin_x, corr_hist = calc_corellation(spike_times, spike_ids, num, duration)

    # Plot rate histogram
    pop_rate_axis = plt.Subplot(fig, gsp[3 - (i / 2), 3 - (i % 2)])
    fig.add_subplot(pop_rate_axis)
    #pop_rate_axis.set_title(name)
    pop_rate_axis.plot(rate_bin_x, rate_hist)
    
    # Plot rate histogram
    pop_cv_isi_axis = plt.Subplot(fig, gsp[3 - (i / 2), 5 - (i % 2)])
    fig.add_subplot(pop_cv_isi_axis)
    #pop_cv_isi_axis.set_title(name)
    pop_cv_isi_axis.plot(isi_bin_x, isi_hist)

    # Plot correlation histogram
    pop_corr_axis = plt.Subplot(fig, gsp[3 - (i / 2), 7 - (i % 2)])
    fig.add_subplot(pop_corr_axis)
    #pop_corr_axis.set_title(name)
    pop_corr_axis.plot(corr_bin_x, corr_hist)

    # Update offset
    start_id += num

#for i in range(2):
#    pop_rate_axes[-1, i].set_xlim((0.0, 20.0))
#    pop_cv_isi_axes[-1, i].set_xlim((0.0, 1.5))

raster_axis.set_xlabel("Time [ms]")
#raster_axis.set_yticks(np.arange(0.0, len(pop_spikes) * 1.0, 1.0))
#main_axes[1].set_yticklabels(zip(*pop_spikes)[2])

#main_axes[1].set_xlabel("Mean firing rate [Hz]")
#

fig.tight_layout()

# Show plot
plt.show()

