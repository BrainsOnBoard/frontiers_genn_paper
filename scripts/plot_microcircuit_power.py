import matplotlib.pyplot as plt
import numpy as np
import plot_settings
import utils

# CSV filename, 'idle' power, connection build time, sim time, spike write time
data = [("microcircuit_power/1050ti.csv", 70.0, 18511.7 + 338.603, 140041, 24039),
        ("microcircuit_power/tx2.csv", 5.5, 541584.0 + 965.12, 258751.0)]


fig, axes = plt.subplots(len(data), figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches))

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

    valid = (trace["power"] < (d[1] * 5.0))
    time = trace["time"][valid]
    power = trace["power"][valid]
    
    # Find point power trace first crossed idle - assume this is experiment start time
    experiment_start_index = np.argmax(power > d[1])
    experiment_start_time = time[experiment_start_index]

    # Make all times relative to experiment start
    time -= experiment_start_time

    # Find index of point where connection generation time has elapsed
    sim_start_index = np.argmax(time > (d[2] / 1000.0))

    exp_end_index = np.argmax(time > ((d[2] + d[3]) / 1000.0))

    # Set title to device
    a.set_title(chr(i + ord("A")), loc="left")

    # Initial idle
    idle_actor = a.fill_between(time[:experiment_start_index],
                                power[:experiment_start_index])


    # Connection building
    conn_build_actor = a.fill_between(time[experiment_start_index:sim_start_index],
                                      power[experiment_start_index:sim_start_index])


    # Simulation
    sim_actor = a.fill_between(time[sim_start_index:exp_end_index],
                               power[sim_start_index:exp_end_index])


    # Final idle
    a.fill_between(time[exp_end_index:],
                   power[exp_end_index:],
                   color=idle_actor.get_facecolor())

    # Calculate mean idle power
    idle_power = np.average(power[:experiment_start_index])

    # Calculate energy to solution
    energy_to_solution = np.trapz(power[experiment_start_index:exp_end_index],
                                  time[experiment_start_index:exp_end_index])

    # **TODO** should we subtract idle power
    sim_energy = np.trapz(power[sim_start_index:exp_end_index],
                          time[sim_start_index:exp_end_index])
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

