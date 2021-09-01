import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import QAgent
from trainer import Trainer
from environment import ExtractionFieldEnv, BaseConfig


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model',
    help='path to numpy file to by loaded into model'
)
parser.add_argument(
    '-t', '--train',
    help='enable training', action='store_true'
)
parser.add_argument(
    '-s', '--save',
    help='path to where model should be saved'
)
parser.add_argument(
    '-c', '--count_states',
    action='store_true',
    help='count states while training (and save state count matrix)'
)
parser.add_argument(
    '-e', '--epochs',
    type=int, nargs='?', const=1,
    help='number of epochs to run, default 1 (ignored if no training)',
)
parser.add_argument(
    '-k', '--show_plot_every',
    type=int,
    help='specify how often to show plot, default 0 (ignored if no training)',
)
args = parser.parse_args()

Q, state_count = None, None
if args.model:
    x = np.load(args.model)
    if args.count_states:
        Q, state_count = x
    else:
        Q = x

epochs = args.epochs


if __name__ == '__main__':
    env = BaseConfig.get_env()

    agent = QAgent(env, [16, 24, 24, 10], Q=Q, use_states=[0,1,2,3])
    n = 288

    if args.train:
        print(f'\nStarting training for {epochs} epochs...\n')
        agent.training(True)
        try:
            state_count = Trainer.n_step_SARSA(
                agent,
                n,
                epochs,
                eps=0.00,
                gamma=0.95,
                show_every=args.show_plot_every,
                save_dest=args.save,
                count_states=args.count_states,
                state_count=state_count
            )
        except KeyboardInterrupt:
            print('\ntraining interrupted')

    if args.save:
        print(f'\nSaving model as {args.save}...')
        if state_count is not None:
            np.save(args.save, [agent.Q, state_count])
        else:
            np.save(args.save, agent.Q)

    print(f'\nEvaluating model...\n')
    agent.training(False)
    rewards = agent.run()

    diffs = np.array([
        max(env.head_history[i:i+60]) - min(env.head_history[i:i+60])
        for i in range(len(rewards)-60)
    ])
    status = 'Time steps {}, mean reward {}, mean diff {}, std diff {}'.format(
        env.t,
        np.round(rewards.mean(), 4),
        np.round(diffs.mean(), 4),
        np.round(diffs.std(), 4),
    )
    print(status)

    Trainer._make_plot(
        agent.env.flow_history,
        [
            agent.env.head_history,
            agent.env.tank_history,
        ],
        ['head', 'tank'],
        ['Head (masl)', 'Tank (m3)'],
    )
    plt.show()
