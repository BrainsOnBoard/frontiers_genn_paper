import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plot_settings
import utils
from scipy.stats import norm

# Load weights
weights = np.fromfile("mad_data/weights.bin", dtype=np.float32)

# Convert weights from nA to pA
weights *= 1000.0

# Calculate weight histogram
min_weight = np.amin(weights)
max_weight = np.amax(weights)
mean_weight = np.average(weights)
std_weight = np.std(weights)
hist, bin_x = np.histogram(weights, bins=np.arange(min_weight, max_weight, 0.75), density=True)

print("Min:%f, max:%f, mean:%f, sd:%f" % (min_weight, max_weight, mean_weight, std_weight))

# Convert bin edges to bin centres
bin_centre_x = bin_x[:-1] + ((bin_x[1:] - bin_x[:-1]) * 0.5)

# Plot histogram
fig, axis = plt.subplots(figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches),
                         frameon=False)

pal = sns.color_palette()
axis.bar(bin_centre_x, hist, width=bin_x[1] - bin_x[0], color=pal[0])

# Plot weight distribution from original paper
#axis.plot(bin_centre_x, norm.pdf(bin_centre_x, mean, std), color=pal[1])
axis.plot(bin_centre_x, norm.pdf(bin_centre_x, 45.65, 3.99), color=pal[1])

axis.set_xlim((30.0, 60.0))
axis.set_xlabel("Weight [pA]")
axis.set_ylabel("Fraction of synapses")

utils.remove_axis_junk(axis)

fig.tight_layout(pad=0.0)
fig.savefig("../figures/mad_weights.eps")
plt.show()
