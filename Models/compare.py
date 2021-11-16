import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

res = pd.read_csv("sim.out", header=None)
a = res.to_numpy()
a = a.flatten()
Q0 = [float(a_i.split()[0]) for a_i in a]
Q1 = [float(a_i.split()[1]) for a_i in a]
x = range(len(Q0))
input_file = "Data\\Data.csv"
data = pd.read_csv(input_file)
precip = data['Rain(mm)'].to_numpy()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('days')
ax1.set_ylabel('Perc', color=color)
ax1.plot(x, precip, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('DREAMS', color=color)  # we already handled the x-label with ax1
ax2.plot(x, Q0, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()