import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.optimize import leastsq, curve_fit

from datetime import datetime
from data_manager import DataManager

# np.random.seed(43)

# prepare data
dm = DataManager('../data/complete/all_no_energy.csv')
df = dm.get(
    'Tryk DP2', 'Dato',
    pumps='all', measurements=['Flow'], as_np=False
).interpolate()
df.rename({'Tryk DP2': 'Head'}, inplace=True, axis=1)
df['Flow'] = df[[f'Flow DP{i}' for i in range(1,5)]].sum(axis=1)
df.set_index('Dato', inplace=True)


# define relevant functions

def season(t, a, b, c, d=0):
    ''' sine based seasonality '''
    return a * np.sin(b * (t + c)) + d

def m3h_to_m3(m3h):
    ''' convert flowrate to volume '''
    return m3h / 12

def vol_to_height(vol, r):
    ''' convert volume to height in meters '''
    return vol / (np.pi * r**2)

def conv(inp):
    mask = np.array([0.09*(0.985**i) for i in range(2*288)])[::-1]
    l = np.array([inp[i:i+mask.shape[0]].dot(mask) for i in range(inp.shape[0]-mask.shape[0])])
    return np.concatenate((inp[:inp.shape[0]-l.shape[0]], l))

def f(X, amp, freq, pha, r):
    ''' combined functions '''
    t, flow = X
    return season(t, amp, freq, pha, -vol_to_height(flow, r))

# define number of data points and initial water level
N = len(df)
wl = df['Head'].iloc[0]

# interpolate over head to get rid of NaNs and subtract mean
head = df['Head'].interpolate()
ydata = head - head.mean()

# build xdata from timeseries (range 1 to N) and converted flowrate
flow = df['Flow'].to_numpy()
xdata = np.array([np.arange(N), m3h_to_m3(flow)])

# guess initial parameters
g_r = 1.5
g_amp = 1.
g_fre = (2*np.pi) / (N / (N / (365 * 288)))
g_pha = 0

# fit season curve
popt, _ = curve_fit(
    season, xdata[0], ydata, p0=[g_amp, g_fre, g_pha]
)

# np.save('./params.npt', popt)

# predict delta_head
# delta_head = f(xdata, *popt, g_r)
# seasonal = season(xdata[0], *popt, d=head.mean())

g_r = 4.5
wl = np.ones((N,)) * 10 + season(xdata[0], *popt)
pred_head = [wl[0] + vol_to_height(flow[0], g_r)]
for t in range(1, N):
    y = wl[t] - vol_to_height(flow[t], g_r)
    ht = pred_head[t-1] + (0.05 * (y - pred_head[t-1]))
    pred_head.append(ht)

# generate noise
mu = 0
sigma = 0.0015
noise = np.random.normal(mu, sigma, (N,)).cumsum()

# make final head prediction (and round to match true head)
# pred_head = np.round((np.ones((N,)) * wl) + delta_head + noise, 2)
pred_head = np.round(np.array(pred_head) + noise, 2)


# plotting
ax, ax2 = plt.gca(), plt.twinx()
ax.plot(df.index, flow, 'b--', label='flow')
ax2.plot(df.index, df['Head'], c='r', label='true head')
ax2.plot(df.index, pred_head, c='g', label='simulated head')
# ax2.plot(df.index, noise, 'y--', label='noise')

# plt.legend()
plt.ylabel('Head (masl)')
handles, labels = [
    (a + b) for a, b in zip(
        ax.get_legend_handles_labels(),
        ax2.get_legend_handles_labels()
    )
]

ax.set_ylabel('Flow (m3/h)')
ax2.set_ylabel('Head (masl)')
plt.legend(handles, labels, loc='upper right')
plt.grid(True)
# plt.savefig(
#     '../report/imgs/envExampleRun1.png',
#     bbox_inches='tight'
# )
plt.show()

def plot2(first, second, sigma=0.1):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(df[first].iloc[378000+(2*288):380100], c='r', label=first)
    ax2.plot(df[second].iloc[378000+(2*288):380100], label=second)

    ax.set_ylabel('Head (masl)')
    ax2.set_ylabel('Flow (m3/h)')

    handles, labels = [
        (a + b) for a, b in zip(
            ax.get_legend_handles_labels(),
            ax2.get_legend_handles_labels()
        )
    ]

    fig.autofmt_xdate()

    plt.legend(handles, labels, loc='upper left')
    plt.grid(True)
    plt.savefig(
        '../report/imgs/steadyStateConvergence.png',
        bbox_inches='tight'
    )
    plt.close()
