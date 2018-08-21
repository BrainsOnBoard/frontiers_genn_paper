import matplotlib.pyplot as plt
import numpy as np
import plot_settings
import utils

# CSV filename, 'idle' power, connection build time, sim time
data = [#("GeForce 1050Ti", (61.3,), (19.032, 106.5), (15.5688, 124.2)),
        ("microcircuit_power/tx2.csv", 5.5, 541584.0 + 965.12, 258751.0)]


fig, axes = plt.subplots(len(data) + 1, figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches))

# How long to plot idle time for
idle_time_s = 10.0

total_synaptic_events = 938037605 * 10

idle_actor = None
conn_build_actor = None
simn_actor = None

# Loop through devices
for i, (d, a) in enumerate(zip(data, axes)):
    # Load trace
    trace = np.loadtxt(d[0], skiprows=1, delimiter=",",
                       dtype={"names": ("time", "power", ), "formats": (float, float)})

    # Find point power trace first crossed idle - assume this is experiment start time
    experiment_start_index = np.argmax(trace["power"] > d[1])
    experiment_start_time = trace["time"][experiment_start_index]

    # Make all times relative to experiment start
    trace["time"] -= experiment_start_time

    # Find index of point where connection generation time has elapsed
    sim_start_index = np.argmax(trace["time"] > (d[2] / 1000.0))

    exp_end_index = np.argmax(trace["time"] > ((d[2] + d[3]) / 1000.0))

    # Set title to device
    a.set_title(chr(i + ord("A")), loc="left")

    # Initial idle
    idle_actor = a.fill_between(trace["time"][:experiment_start_index],
                                trace["power"][:experiment_start_index])


    # Connection building
    conn_build_actor = a.fill_between(trace["time"][experiment_start_index:sim_start_index],
                                      trace["power"][experiment_start_index:sim_start_index])


    # Simulation
    sim_actor = a.fill_between(trace["time"][sim_start_index:exp_end_index],
                               trace["power"][sim_start_index:exp_end_index])


    # Final idle
    a.fill_between(trace["time"][exp_end_index:],
                   trace["power"][exp_end_index:],
                   color=idle_actor.get_facecolor())

    # Calculate mean idle power
    idle_power = np.average(trace["power"][:experiment_start_index])

    # Calculate energy to solution
    energy_to_solution = np.trapz(trace["power"][experiment_start_index:exp_end_index],
                                  trace["time"][experiment_start_index:exp_end_index])

    # **TODO** should we subtract idle power
    sim_energy = np.trapz(trace["power"][sim_start_index:exp_end_index],
                          trace["time"][sim_start_index:exp_end_index])
    energy_per_synaptic_event = sim_energy/ float(total_synaptic_events)

    print("%s:" % (d[0]))
    print("\tIdle power = %fW" % (idle_power))
    print("\tEnergy to solution = %fJ" % (energy_to_solution))
    print("\tEnergy per synaptic event = %fuJ" % (energy_per_synaptic_event * 1E6))

    a.axvline(0.0, color="black", linestyle="--")
    a.axvline((d[2] + d[3]) / 1000.0, color="black", linestyle="--")
    a.set_xlabel("Simulation time [s]")
    a.set_ylabel("Power [W]")

fig.legend([idle_actor, conn_build_actor, sim_actor], ["Idle", "Initialisation", "Simulation"],
           loc="lower center", ncol=3)
fig.tight_layout(pad=0.0)
fig.savefig("../figures/microcircuit_power.eps")
plt.show()

