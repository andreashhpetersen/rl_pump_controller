import gym
import numpy as np
import pandas as pd

from gym import Env, spaces
from datetime import datetime, timedelta

PREV_FLOW = 0
HEAD = 1
RUNNING_MEAN = 2
TANKS = 3
DAY = 4
HOUR = 5


class ExtractionFieldEnv(Env):
    def __init__(
        self, init_wl, init_flow, amp, freq, pha, r, sigma, c, d,
        reward_model_type='head',
        start_ts=0, run_for_ts=526176, ts_size=5, start_year=2016
    ):

        self.init_wl = init_wl
        self.init_flow = init_flow
        self.init_tank = 1000.
        self.start_ts = start_ts

        self.amp = amp
        self.freq = freq
        self.pha = pha
        self.r = r
        self.sigma = sigma
        self.c = c
        self.d = d

        self.t = start_ts
        self.ts_size = ts_size
        self.run_for_ts = run_for_ts
        self.ts_in_hour = 60 // ts_size
        self.ts_in_day = self.ts_in_hour * 24
        self.ts_in_year = self.ts_in_day * 365
        self.ws = self.ts_in_hour * 5
        self.start_year = start_year

        self.consumption = self._init_consumption_matrix()

        self.HEAD_HARD_MIN = 7.
        self.MAX_TANK = 2000.
        self.MAX_HEAD = 10.83
        self.MAX_PENALTY = -9999

        if not reward_model_type in ['head', 'flow']:
            raise ValueError(f'''
                keyword argument `reward_model_type` must be either `head` or
                `flow`, not `{reward_model_type}`
                ''')
        else:
            self.reward_model_type = reward_model_type

        # observations (flow, head, roc, tank vol, month, day, hour)
        self.observation_space = spaces.Box(
            low=np.array([45., self.HEAD_HARD_MIN, -10., 0, 1, 0, 0]),
            high=np.array([125., 12., 10., self.MAX_TANK, 13, 7, 24]),
            dtype=np.float64
        )

        self.action_space = spaces.Discrete(16)

    def step(self, flow):
        self.t += 1

        self.prev_flow = self.flow
        self.flow = flow
        self.water_in += flow / self.ts_in_hour

        self._update_wl()
        self._update_head()
        self._update_tank_vol()
        self._update_running_head_mean()

        self.flow_history.append(self.flow)
        self.tank_history.append(self.tank_vol)
        return self.state, self.reward, self._terminal()

    def reset(self):
        if self.start_ts is None:
            self.t = np.random.randint(0, self.ts_in_year)
        else:
            self.t = self.start_ts

        self.t0 = self.t
        self.t0_datetime = \
            datetime(self.start_year, 1, 1) + timedelta(self.t0 * self.ts_size)

        self.flow = self.init_flow
        self.prev_flow = self.init_flow

        self.wl = self.init_wl

        self.tank_vol = self.init_tank
        self.water_in = 0
        self.water_out = 0

        h0 = self.wl + self.seasonal_component() - self.flow_component()
        self.head_history = [h0]
        self.flow_history = [self.init_flow]
        self.tank_history = [self.init_tank]

        self.running_head_sum = self.head
        self.running_head_mean = self.head

    @property
    def state(self):
        _, d, h = self.t_to_month_weekday_hour()
        return np.array(
            [
                self.flow,
                min(self.head, self.MAX_HEAD),
                self.running_head_mean,
                min(self.tank_vol, self.MAX_TANK-1),
                d,
                h,
            ]
        )

    @property
    def reward(self):
        # calculate reward for head
        if self.head < self.HEAD_HARD_MIN:
            h = self.MAX_PENALTY
        elif self.reward_model_type == 'head':
            h = -np.square(1 + np.abs(self.running_head_mean - self.head))
        else:
            h = 0

        # calculate reward for tank volume
        t = 0
        if self.tank_vol < 0:
            t = self.MAX_PENALTY
        elif self.tank_vol > self.MAX_TANK:
            t = self.MAX_PENALTY

        # we don't use flow reward right now
        f = 0
        if self.flow != self.prev_flow:
            if self.reward_model_type == 'flow':
                if self.flow != 0:
                    f = -((np.abs(self.flow - self.prev_flow) / 5)**1.5)
            else:
                f = -0.5

        return h + f + t

    @property
    def head(self):
        return self.head_history[-1]

    def seasonal_component(self):
        sine = np.sin(self.freq * (self.t + self.pha))
        return self.amp * sine

    def flow_component(self):
        vol = self.flow / self.ts_in_hour
        return vol / (np.pi * self.r**2)

    def _update_head(self):
        """
        calculates h_t = h_{t-1} + d * (w_t - (a*f_t / r * pi^2) - h_{t-1})
        and appends it to head history
        """
        wl = self.wl + self.seasonal_component()
        prev_h = self.head_history[-1]
        new_h = prev_h + self.d * (wl - self.flow_component() - prev_h)
        self.head_history.append(new_h)

    def _update_running_head_mean(self):
        self.running_head_sum += self.head
        if (self.t - self.t0) <= self.ws:
            self.running_head_mean = self.running_head_sum / (self.t - self.t0 + 1)
        else:
            self.running_head_sum -= self.head_history[-self.ws]
            self.running_head_mean = self.running_head_sum / self.ws

    def t_to_month_weekday_hour(self):
        """ convert timestep to datetime and return (month, weekday, hour) """
        dt = self.t0_datetime + timedelta(minutes=self.t * self.ts_size)
        return (int(dt.month), int(dt.weekday()), int(dt.hour))

    def _terminal(self):
        """ test if we have reached terminal state """
        return (
            self.t > (self.t0 + self.run_for_ts) or
            self.head < self.HEAD_HARD_MIN or
            self.tank_vol < 0 or
            self.tank_vol > self.MAX_TANK
        )

    def _update_wl(self):
        noise = np.random.normal(0, self.sigma)
        self.wl += self.c - self._out_flow() + noise

    def _update_tank_vol(self):
        _, d, h = self.t_to_month_weekday_hour()
        self.tank_vol += self.flow / self.ts_in_hour
        cons = self.consumption[d][h]
        demand = cons + np.random.normal(scale=self.cons_std)
        self.tank_vol -= demand
        self.water_out += demand

    def _out_flow(self):
        vol = self.flow / self.ts_in_hour
        return vol * 2.1494109589041096e-07

    def _init_consumption_matrix(self):
        df = pd.read_csv(
            '../data/complete/ConsumptionProfilesUnNormalized.csv',
            delimiter=';', decimal=r','
        )
        cons = df.groupby(['WeekdayNumber', 'Hour']).sum()['Consumption']
        cons = cons.to_numpy().reshape(7, 24)
        cons /= self.ts_in_hour
        cons *= 3.2
        self.cons_std = cons.std()
        return cons


class BaseConfig:
    @classmethod
    def get_env(cls, **kwargs):
        amp, freq, pha = np.load('./params.npy')
        r = 4.5
        c = 0.1 / (365 * 288)    # water rises 10 cm pr year
        d = 0.05
        sigma = 0.0015
        start_ts = None   # makes env choose randomly
        run_for_ts = 288 * 365
        init_wl = 9.11
        init_flow = 80

        ENV_CONFIG = {
            'amp': amp,
            'freq': freq,
            'pha': pha,
            'r': r,
            'd': d,
            'sigma': sigma,
            'c': c,
            'init_wl': init_wl,
            'init_flow': init_flow,
            'start_ts': start_ts,
            'run_for_ts': run_for_ts,
        }

        for k, v in kwargs.items():
            ENV_CONFIG[k] = v

        env = ExtractionFieldEnv(**ENV_CONFIG)
        return env
