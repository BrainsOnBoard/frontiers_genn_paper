import matplotlib.pyplot as plt
import numpy as np
import plot_settings
import utils
from scipy.stats import norm

# Load weights
weights = np.fromfile("mad_data/weights.bin", dtype=np.float32)

# Convert weights from nA to pA
weights *= 1000.0

# Calculate weight histogram
hist, bin_x = np.histogram(weights, bins=40, range=(30.0, 60.0))

# Normalise
hist = np.divide(hist, len(weights), dtype=float)

# Convert bin edges to bin centres
bin_centre_x = bin_x[:-1] + ((bin_x[1:] - bin_x[:-1]) * 0.5)

# Plot histogram
fig, axis = plt.subplots(figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches),
                         frameon=False)
axis.bar(bin_centre_x, hist, width=bin_x[1] - bin_x[0])

axis.plot(bin_centre_x, norm.pdf(bin_centre_x, 45.65, 3.99))
fig.tight_layout(pad=0.0)
plt.show()
