import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from data_manager import DataManager

dm = DataManager('../data/complete/all_no_energy.csv')
df = dm.default_df()
heads = df.Head.to_numpy()

ws = 60
diffs = np.array([
    np.square(1 + np.abs(np.mean(heads[i-ws:i]) - heads[i]))
    for i in range(ws, len(heads))
])
diffs = np.concatenate((np.array([None]*ws), diffs))

closer = np.logical_and(
    df.index >= datetime(2019, 8, 5),
    df.index <= datetime(2019, 8, 17)
)

fig, axs = plt.subplots(2, 1)
fig.suptitle('Head and (positive) reward')

ax1, ax2 = axs[0], axs[0].twinx()
lns1 = ax1.plot(df.index, heads, c='r', label='Head')
lns2 = ax2.plot(df.index, diffs, c='black', label='Penalty')
ax1.grid(True)
lns = lns1 + lns2
ax1.legend(lns, [ln.get_label() for ln in lns])
ax1.tick_params(labelrotation=45)
ax1.set_ylabel('Head (masl)')
ax2.set_ylabel('Penalty (positive)')

ax1, ax2 = axs[1], axs[1].twinx()
lns1 = ax1.plot(
    df.index[closer],
    heads[closer],
    c='r',
    label='Head'
)
lns2 = ax2.plot(
    df.index[closer],
    diffs[closer],
    c='black',
    label='Penalty'
)
ax1.grid(True)
lns = lns1 + lns2
ax1.legend(lns, [ln.get_label() for ln in lns])
ax1.tick_params(labelrotation=45)
ax1.set_ylabel('Head (masl)')
ax2.set_ylabel('Penalty (positive)')

plt.tight_layout()
fig.savefig(
    '../report/imgs/headByReward.png',
    # bbox_inches='tight'
)
# plt.show()
