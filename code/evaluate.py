import numpy as np
import matplotlib.pyplot as plt

from models import QAgent
from trainer import Trainer
from environment import ExtractionFieldEnv, BaseConfig
from plot_state_count import tank_flow_bar_plot, head_vs_mean_head_bar_plot

def evaluate(agent, episodes=100, verbose=True):
    agent.training(False)

    env = agent.env
    timesteps, rewards, high_diffs = [], [], []
    flow_changes, flow_change_sizes = [], []
    water_in, water_out = [], []
    terminations = []

    # reasons for terminating
    NOT_DONE = 0
    OVERFLOW = 1
    UNDERFLOW = 2
    HEAD = 3

    for ep in range(episodes):
        env.reset()
        rs = agent.run()

        T = env.t - env.t0
        timesteps.append(min(T, env.run_for_ts))
        rewards.append(np.mean(rs))

        if env.tank_vol < 0:
            terminations.append(UNDERFLOW)
        elif env.tank_vol > env.MAX_TANK:
            terminations.append(OVERFLOW)
        elif env.head < env.HEAD_HARD_MIN:
            terminations.append(HEAD)
        else:
            terminations.append(NOT_DONE)

        flow_change = 0
        diffs, change_sizes = [], []
        heads, flows = env.head_history, env.flow_history
        for i in range(len(heads)):
            if i >= 60:
                diffs.append(np.abs(np.mean(heads[i-60:i]) - heads[i]))

            if flows[i] != flows[i-1]:
                flow_change += 1
                change_sizes.append(np.abs(flows[i] - flows[i-1]))

        diffs = np.sort(diffs)
        high_diffs.append(np.mean(diffs[-int(0.1 * len(diffs)):]))

        flow_changes.append(flow_change / (len(flows)-1))
        flow_change_sizes.append(np.mean(change_sizes))

        water_in.append(env.water_in)
        water_out.append(env.water_out)

        if verbose:
            status = [
                ('Time steps', np.round(timesteps[-1], 2)),
                ('Reward', np.round(rewards[-1], 2)),
                ('# flow change', np.round(flow_changes[-1], 2)),
                ('head change', np.round(high_diffs[-1], 2)),
            ]
            print(f'Episode {ep + 1}')
            print(('{:<20}'*len(status)).format(*[
                f'{t}: {v}' for t,v in status
            ]))

    data = np.array([
        timesteps,
        rewards,
        high_diffs,
        flow_changes,
        flow_change_sizes,
        water_in,
        water_out,
        terminations
    ])
    return data.T



if __name__ == '__main__':
    MODELS_DIR = './saved_models/'
    OUTPUT_DIR = './output/'
    models = [
        ('headModel.npy', 'headModelLongRunEval.npy', 'head'),
        ('headModel_longEps.npy', 'headModel_longEpsLongRunEval.npy', 'head'),
        ('flowModel.npy', 'flowModelLongRunEval.npy', 'flow'),
        ('flowModel_longEps.npy', 'flowModel_longEpsLongRunEval.npy', 'flow'),
    ]

    for name, output_fn, typ in models:
        print(f'\nEvaluating {name}...\n')
        Q, C = np.load(MODELS_DIR + name)

        env = BaseConfig.get_env(
            run_for_ts=10 * (365 * 288),
            reward_model_type=typ
        )
        if typ == 'head':
            states = [0,1,2,3]
        else:
            states = [0,3,4,5]

        agent = QAgent(env, Q.shape[:-1], Q=Q, use_states=states)
        data = evaluate(agent, episodes=100)
        np.save(OUTPUT_DIR + output_fn, data)
