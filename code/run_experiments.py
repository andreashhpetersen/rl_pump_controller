import numpy as np
import matplotlib.pyplot as plt

from models import QAgent
from trainer import Trainer
from environment import BaseConfig

PREV_FLOW = 0
HEAD = 1
RUNNING_MEAN = 2
TANKS = 3
DAY = 4
HOUR = 5


if __name__ == '__main__':
    env = BaseConfig.get_env()

    Q,state_count = np.load('./saved_models/headModel_longEps.npy')
    # Q = None

    agent = QAgent(env, [16, 24, 24, 10], Q=Q, use_states=[
        PREV_FLOW,HEAD,RUNNING_MEAN,TANKS
    ])
    # state_count = np.zeros(agent.Q.shape)

    # important to specify these and not override some existing shit!
    save_as = './saved_models/headModel_longEps.npy'
    save_log_fn = './output/headModelTrainingLog_longEps.csv'

    n = 288
    for epochs, eps in [(1432, 0.0)]:
        print(f'\nStarting training for {epochs} epochs with epsilon {eps}...\n')
        agent.training(True)
        try:
            state_count = Trainer.n_step_SARSA(
                agent,
                n,
                epochs,
                eps=eps,
                gamma=0.95,
                save_log_fn=save_log_fn,
                count_states=True,
                state_count=state_count
            )
        except KeyboardInterrupt:
            print('\ntraining interrupted')

        print(f'\nSaving model as {save_as}...')
        if state_count is not None:
            np.save(save_as, [agent.Q, state_count])
        else:
            np.save(save_as, agent.Q)
