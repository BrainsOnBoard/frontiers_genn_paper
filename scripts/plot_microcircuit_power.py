import matplotlib.pyplot as plt
import numpy as np
import plot_settings
import utils

# CSV filename, 'idle' power, sim time, spike write time
data = [("microcircuit_power/k40c.csv", 150.0, 41911.5, 6199.62),
        ("microcircuit_power/1050ti.csv", 70.0, 137592, 15054),
        ("microcircuit_power/tx2.csv", 6.0, 258350, 14516.2)]


fig, axes = plt.subplots(len(data), figsize=(plot_settings.column_width, 90.0 * plot_settings.mm_to_inches))

# How long to plot idle time for
idle_time_s = 10.0

total_synaptic_events = 938037605 * 10

idle_actor = None
init_actor = None
sim_actor = None
spike_write_actor = None

# Loop through devices
for i, (d, a) in enumerate(zip(data, axes)):
    # Load trace
    trace = np.loadtxt(d[0], skiprows=1, delimiter=",",
                       dtype={"names": ("time", "power", ), "formats": (float, float)})

    # Filter out clearly erroneous values
    valid = (trace["power"] < (d[1] * 5.0))
    time = trace["time"][valid]
    power = trace["power"][valid]
    
    # Find points power trace first crossed idle - assume first and last are experiment start and end times
    exp_non_idle_indices = np.where(power > d[1])[0]
    exp_start_index = exp_non_idle_indices[0]
    exp_end_index = exp_non_idle_indices[-1]
    exp_start_time = time[exp_start_index]
    exp_end_time = time[exp_end_index]

    sim_end_time = exp_end_time - (d[3] / 1000.0)
    sim_end_index = np.argmax(time > sim_end_time)

    sim_start_time = sim_end_time - (d[2] / 1000.0)
    sim_start_index = np.argmax(time > sim_start_time)

    # Make all times relative to experiment start
    time -= exp_start_time

    # Set title to device
    a.set_title(chr(i + ord("A")), loc="left")

    # Initial idle
    idle_actor = a.fill_between(time[:exp_start_index],
                                power[:exp_start_index])


    # Connection building
    init_actor = a.fill_between(time[exp_start_index - 1:sim_start_index],
                                power[exp_start_index - 1:sim_start_index])


    # Simulation
    sim_actor = a.fill_between(time[sim_start_index - 1:sim_end_index],
                               power[sim_start_index - 1:sim_end_index])


    # Spike writing
    spike_write_actor = a.fill_between(time[sim_end_index - 1:exp_end_index],
                               power[sim_end_index - 1:exp_end_index])
    # Final idle
    a.fill_between(time[exp_end_index - 1:],
                   power[exp_end_index - 1:],
                   color=idle_actor.get_facecolor())

    # Calculate mean idle power
    idle_power = np.average(np.hstack((power[:exp_start_index], power[exp_end_index:])))

    # Calculate energy to solution
    energy_to_solution = np.trapz(power[exp_start_index:exp_end_index],
                                  time[exp_start_index:exp_end_index])

    # **TODO** should we subtract idle power
    sim_energy = np.trapz(power[sim_start_index:exp_end_index],
                          time[sim_start_index:exp_end_index])
    energy_per_synaptic_event = sim_energy/ float(total_synaptic_events)

    print("%s:" % (d[0]))
    print("\tIdle power = %fW" % (idle_power))
    print("\tEnergy to solution = %fJ = %fkWh" % (energy_to_solution, energy_to_solution / 3600000.0))
    print("\tSimulation energy = %fJ = %fkWh" % (sim_energy, sim_energy / 3600000.0))
    print("\tEnergy per synaptic event = %fuJ" % (energy_per_synaptic_event * 1E6))

    a.axvline(0.0, color="black", linestyle="--")
    a.axvline(exp_end_time - exp_start_time, color="black", linestyle="--")
    a.set_xlabel("Simulation time [s]")
    a.set_ylabel("Power [W]")

fig.legend([idle_actor, init_actor, sim_actor, spike_write_actor],
           ["Idle", "Initialisation", "Simulation", "Spike writing"],
           loc="lower center", ncol=2)
fig.tight_layout(pad=0.0, rect=(0.0, 0.125, 1.0, 1.0))
fig.savefig("../figures/microcircuit_power.eps")
plt.show()

