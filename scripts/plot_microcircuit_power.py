import matplotlib.pyplot as plt
import plot_settings
import utils

# Meter power, GPU power, CPU power
data = [("GeForce 1050Ti", (61.3,), (19.032, 106.5), (15.5688, 124.2)),
        ("Jetson TX2", (5.3, 0.2, 0.305), (542.632, 6.1, 0.2, 0.7), (25.7645, 10.8, 2.061, 0.916))]


fig, axes = plt.subplots(len(data), figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches))

# How long to plot idle time for
idle_time_s = 10.0

total_synaptic_events = 938037605

idle_actor = None
conn_build_actor = None
simn_actor = None

# Loop through devices
for d, a in zip(data, axes):
    # Set title to device
    a.set_title(d[0])

    t = [-idle_time_s, 0.0, d[2][0], d[2][0] + d[3][0], d[2][0] + d[3][0] + idle_time_s]

    p_total = [d[1][0], d[2][1], d[3][1], d[1][0], 0]

    # Initial idle
    idle_actor = a.fill_between([-idle_time_s, 0.0], [d[1][0], d[1][0]])

    # Connection building
    conn_build_actor = a.fill_between([0.0, d[2][0]], [d[2][1], d[2][1]])

    # Simulation
    sim_actor = a.fill_between([d[2][0], d[2][0] + d[3][0]], [d[3][1], d[3][1]])

    # Final idle
    a.fill_between([d[2][0] + d[3][0], d[2][0] + d[3][0] + idle_time_s], [d[1][0], d[1][0]], color=idle_actor.get_facecolor())

    # Calculate energy to solution
    energy_to_solution = (d[2][0] * d[2][1]) + (d[3][0] * d[3][1])
    energy_per_synaptic_event = d[3][0] * d[3][1] / float(total_synaptic_events)

    print("%s:" % (d[0]))
    print("\tEnergy to solution = %fJ" % (energy_to_solution))
    print("\tEnergy per synaptic event = %fuJ" % (energy_per_synaptic_event * 1E6))

    a.axvline(0.0, color="black", linestyle="--")
    a.axvline(d[2][0] + d[3][0], color="black", linestyle="--")
    a.set_xlabel("Simulation time [s]")
    a.set_ylabel("Power [W]")

fig.tight_layout(pad=0.0)

plt.show()

