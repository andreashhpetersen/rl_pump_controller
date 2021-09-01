import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from data_manager import DataManager
from environment import ExtractionFieldEnv, PREV_FLOW, HEAD, RUNNING_MEAN, \
    TANKS, DAY, HOUR


class QAgent:
    def __init__(self, env, discretization,
                 use_states=None, Q=None, lr=0.1, d=1.):
        self.env = env
        self.lr = lr
        self.d = d
        if use_states is None:
            self.use_states = [
                PREV_FLOW, HEAD, RUNNING_MEAN, TANKS, DAY, HOUR
            ]
        else:
            self.use_states = use_states

        # map from discrete action to flow value
        self.ACTION_MAP = {
            0: 0,
            1: 50,
            2: 55,
            3: 60,
            4: 65,
            5: 70,
            6: 75,
            7: 80,
            8: 85,
            9: 90,
            10: 95,
            11: 100,
            12: 105,
            13: 110,
            14: 115,
            15: 120
        }

        # load or initialise Q table
        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.random.uniform(
                -.1, .1, discretization + [len(self.ACTION_MAP)]
            )

    def get_discrete_state(self, state):
        dstate = []

        for s in self.use_states:
            if s == PREV_FLOW:
                # flow state
                if state[PREV_FLOW] == 0:
                    state[PREV_FLOW] = 45
                dstate.append(int(state[PREV_FLOW] / 5 - 9))

            if s == HEAD:
                dstate.append(int((state[HEAD] - 7) / 0.16))

            if s == RUNNING_MEAN:
                dstate.append(int((state[RUNNING_MEAN] - 7) / 0.16))

            if s == TANKS:
                dstate.append(int(state[TANKS] / 200))

            if s == DAY:
                dstate.append(int(state[DAY]))

            if s == HOUR:
                dstate.append(int(state[HOUR]))

        return tuple(dstate)

    def pick_action(self, state, e=0.):
        if np.random.random() < e:
            return int(np.random.random() * len(self.ACTION_MAP))
        return np.argmax(self.Q[state])

    def run(self, verbose=False, C=None):
        """
        Run the agent for an episode without performing any policy improvement.
        If `C` is provided, it will be used to store count of every
        state/action-pair visited
        """
        self.env.reset()
        discrete_state = self.get_discrete_state(self.env.state)

        rewards = []
        done = False
        while not done:
            action = self.pick_action(discrete_state)
            if C is not None:
                C[discrete_state + (action,)] += 1
            new_state, reward, done = self.env.step(self.ACTION_MAP[action])
            rewards.append(reward)

            new_discrete_state = self.get_discrete_state(new_state)

            if self.train:
                current_q = self.Q[discrete_state + (action,)]
                max_future_q = np.max(self.Q[new_discrete_state])

                current = (1) * current_q
                future = self.lr * (reward + self.d * max_future_q - current_q)
                new_q = current + future
                self.Q[discrete_state + (action,)] = new_q

            discrete_state = new_discrete_state

        if verbose:
            print("I've got nothing to say to you!")
            return np.array(rewards)

        if C is not None:
            return np.array(rewards), C
        return np.array(rewards)

    def training(self, train=True):
        self.train = train
