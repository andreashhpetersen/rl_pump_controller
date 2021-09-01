import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from data_manager import DataManager

dm = DataManager('../data/complete/all_no_energy.csv')
df = dm.default_df()
df = df[np.logical_and(df.index > datetime(2019, 8, 7), df.index < datetime(2019, 8, 13))]

wl = (np.ones((14*288,)) * 9.8) + np.random.normal(0, 0.01, (14*288,))
flow = df['Flow'].to_numpy()

f = lambda x: (x / 12) / (np.pi * 1.2**2)

d = 0.03
head = [wl[0] - f(flow[0])]
for t in range(1, len(flow)):
    y = wl[t] - f(flow[t])
    ht = head[t-1] + (d * (y - head[t-1]))
    head.append(ht)


ticks, labels = [], []
for i in range(0, len(flow), 288):
    ticks.append(i)
    labels.append(f'Day {int(i/288) + 1}')

ax, ax2 = plt.gca(), plt.twinx()
ax.plot(head, 'g', label='my head')
ax.plot(df['Head'].to_numpy(), 'r', label='true head')
ax2.plot(flow, 'b')
ax.set_xticks(ticks)
ax.set_xticklabels(labels, rotation=90)
plt.grid(True)
plt.show()
plt.close()
