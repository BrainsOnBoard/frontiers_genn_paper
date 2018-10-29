import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import plot_settings
import utils

scales = np.asarray([1.0, 0.75, 0.5, 0.25])
data = [("Tesla K40c", np.asarray([41911.5, 33950.1, 26541.2, 20955.2])),
        ("Tesla V100", np.asarray([21645.4,20156, 17562, 15512.3])),
        ("GeForce 1050ti", np.asarray([137592, 119623, 102452, 87691.1])),
        ("Jetson TX2", np.asarray([258350, 180802, 110047, 51710.6]))]

fig, axis = plt.subplots(figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches),
                         frameon=False)

actors = [axis.plot(scales * 77169, times / 1000.0, marker="x")[0] for _, times in data]

axis.set_xlabel("Number of neurons")
axis.set_ylabel("Time [s]")
axis.axhline(10.0, color="black", linestyle="--")

fig.legend(actors, [n for n, _ in data],
           loc="lower center", ncol=2)
fig.tight_layout(pad=0, rect=[0.0, 0.15, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/microcircuit_scaling.eps")
plt.show()