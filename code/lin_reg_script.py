import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

from data_manager import DataManager

fn_data = '../data/complete/all_no_energy.csv'
dm = DataManager(fn_data)
df = dm.default_df(start=datetime(2017, 1, 1))

print(df.head())

df['head_lag_12'] = df['Head'].shift(12)
df['flow_lag_12'] = df['Flow'].shift(12)
df.dropna(inplace=True)
df['delta_head_12'] = df['Head'] - df['head_lag_12']
df['delta_flow_12'] = df['Flow'] - df['flow_lag_12']

fig, ax = plt.subplots(2, 1)
fig.suptitle('Rate of change in P vs H')

model = LinearRegression()

model.fit(df.delta_flow.values.reshape(-1, 1), df.delta_head)
domain = np.arange(df.delta_flow.min(), df.delta_flow.max())
fx = model.intercept_ + model.coef_ * domain
ax[0].scatter(df.delta_flow, df.delta_head, c='b')
ax[0].plot(domain, fx, c='r')
ax[0].set_ylabel('Delta H')
ax[0].set_title('Lag 1')

model.fit(df.delta_flow_12.values.reshape(-1, 1), df.delta_head_12)
domain = np.arange(df.delta_flow_12.min(), df.delta_flow_12.max())
fx = model.intercept_ + model.coef_ * domain
ax[1].scatter(df.delta_flow_12, df.delta_head_12, c='b')
ax[1].plot(domain, fx, c='r')
ax[1].set_xlabel('Delta P')
ax[1].set_ylabel('Delta H')
ax[1].set_title('Lag 12')

ax[0].grid(True, which='both', axis='both')
ax[1].grid(True, which='both', axis='both')
plt.tight_layout()
# plt.show()
plt.savefig('../report/imgs/linRegOnRateOfChange.png')
