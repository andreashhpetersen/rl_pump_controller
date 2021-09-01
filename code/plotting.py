import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

from data_manager import DataManager

fn_data = '../data/complete/all_no_energy.csv'
dm = DataManager(fn_data)


df = dm.get('Dato', pumps='all', measurements=['Flow', 'Tryk'], as_np=False)
df.set_index('Dato', inplace=True)
df['Total Flow'] = df[[f'Flow DP{i}' for i in range(1, 5)]].sum(axis=1)
df[df['Total Flow'] == 0] = np.nan

df[['Total Flow'] + [f'Flow DP{i}' for i in range(1, 5)]].plot()

plt.ylabel('m3/h')
plt.legend(loc='upper left')
plt.grid(True)

plt.savefig(
    '../report/imgs/flowAllPumpsAndTotalFlow.png',
    bbox_inches='tight',
)

tryk2head = { f'Tryk DP{i}': f'Head DP{i}' for i in range(1, 5) }
df.rename(tryk2head, inplace=True, axis=1)
cols = [f'Head DP{i}' for i in range(1, 5)]
df[df[cols] < 7] = np.nan
df[cols].plot()

plt.ylabel('Meters above sea level')
plt.legend(loc='upper right')
plt.grid(True)

plt.savefig(
    '../report/imgs/headAllPumps.png',
    bbox_inches='tight'
)
