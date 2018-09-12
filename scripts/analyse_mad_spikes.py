
import numpy as np
from elephant.statistics import isi, cv, fanofactor
from pandas import read_csv

num_excitatory = 90000

print("Loading...")
spikes = read_csv("mad_data/spikes.csv", header=None, names=["time", "id"], skiprows=1, delimiter=",",
                  dtype={"time":float, "id":int})

# Convert CSV columns to numpy
#spike_times = spikes["time"]
#spike_neuron_id = spikes["id"]

min_ms = np.floor(np.amin(spikes["time"]))
max_ms = np.ceil(np.amax(spikes["time"]))

rate, _ = np.histogram(spikes["id"], bins=range(num_excitatory + 1))
assert len(rate) == num_excitatory
mean_rate = np.divide(rate, (max_ms - min_ms) / 1000.0, dtype=float)
print("Mean firing rate: %fHz" % np.average(mean_rate))


# Sort spikes by id
neuron_spikes = spikes.groupby("id")

# Loop through neurons
cv_isi = []
for n in range(num_excitatory):
    try:
        # Get this neuron's spike times
        neuron_spike_times = neuron_spikes.get_group(n)["time"].values

        # If this neuron spiked more than once i.e. it is possible to calculate ISI!
        if len(neuron_spike_times) > 1:
            cv_isi.append(cv(isi(neuron_spike_times)))
    except KeyError:
        pass

print("Mean CV ISI: %f" % np.average(cv_isi))


# Pick 1000 neurons
binned_spike_times = None
for i, n in enumerate(np.random.choice(num_excitatory, 1000, replace=False)):
    # Get this neuron's spike times
    neuron_spike_times = neuron_spikes.get_group(n)["time"].values
    
    # Bin spike times
    neuron_binned_spike_times, _ = np.histogram(neuron_spike_times, bins=np.arange(min_ms, max_ms, 3))
    if binned_spike_times is None:
        binned_spike_times = neuron_binned_spike_times
    else:
        binned_spike_times += neuron_binned_spike_times

# Calculate mean and variance of spike count
mean_spike_count = np.average(binned_spike_times)
var_spike_count = np.var(binned_spike_times)
print("Fano factor: %f" % (var_spike_count / mean_spike_count))
