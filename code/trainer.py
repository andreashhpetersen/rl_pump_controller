import csv
import time
import logging
import numpy as np
import matplotlib.pyplot as plt


class Trainer:

    @classmethod
    def n_step_SARSA(
            self, agent, n, epochs, gamma=0.95, eps=0, lr=0.01,
            show_every=None, save_every=None, save_dest=None,
            save_log_fn='./output/trainingLogData.csv',
            count_states=False, state_count=None
    ):
        if show_every is None:
            show_every = epochs + 1

        if save_every is None:
            save_every = epochs + 1

        if count_states:
            if state_count is not None:
                count = state_count
            else:
                count = np.zeros(agent.Q.shape, dtype=int)

        quantiles = []
        rewards = []
        timesteps = []
        states = []
        pairs = []
        flow_changes = []
        flow_change_sizes = []

        logging.basicConfig(
            filename=save_log_fn, level=logging.DEBUG, format=''
        )

        env = agent.env
        for epoch in range(epochs):
            env.reset()
            A, S, R = [], [], []
            s0 = agent.get_discrete_state(env.state)
            action = agent.pick_action(s0)

            if count_states:
                count[s0 + (action,)] += 1

            S.append(s0)
            A.append(action)
            R.append(0)
            T = np.inf
            for t in range(env.run_for_ts):
                if t < T:
                    new_state, reward, done = env.step(agent.ACTION_MAP[action])

                    state = agent.get_discrete_state(new_state)
                    S.append(state)
                    R.append(reward)

                    if count_states:
                        count[state + (action,)] += 1

                    if done:
                        T = t + 1
                    else:
                        action = agent.pick_action(state, e=eps)
                        A.append(action)

                tao = t - n + 1
                if tao >= 0:
                    steps = np.arange(tao + 1, min(tao + n, T + 1))
                    G = np.sum(
                        gamma**(steps - tao - 1) * np.array(
                            R[steps[0]:steps[-1]+1]
                        )
                    )

                    if tao + n < T:
                        G += (gamma**n) * agent.Q[S[tao+n] + (A[tao+n],)]

                    q_idx = S[tao] + (A[tao],)
                    agent.Q[q_idx] += lr * (G - agent.Q[q_idx])

                if tao == T - 1:
                    break

            # printing and plotting
            heads = env.head_history
            diffs = np.array([
                np.abs(np.mean(heads[i:i+60]) - heads[i])
                for i in range(len(R)-60)
            ])

            timesteps.append(env.t - env.t0)
            rewards.append(np.mean(R))
            quantiles.append(
                np.quantile(diffs, 0.75) if len(diffs) > 0 else None
            )

            flows = env.flow_history
            count_changes = 0
            change_sizes = []
            for i in range(1, len(flows)):
                if flows[i] != flows[i-1]:
                    count_changes += 1
                    change_sizes.append(np.abs(flows[i] - flows[i-1]))

            flow_changes.append(count_changes / (len(flows)-1))
            flow_change_sizes.append(np.mean(change_sizes))

            status = [
                ('Time steps', timesteps[-1], True),
                ('Reward', rewards[-1], True),
                ('Quantile', quantiles[-1], False),
                ('# flow change', flow_changes[-1], True),
                ('mean change', flow_change_sizes[-1], True),
            ]


            if count_states:
                states.append(
                    np.sum(
                        count.reshape(-1, count.shape[-1]).sum(axis=1) != 0
                    )
                )
                pairs.append(np.sum(count != 0))
                status += [
                    ('States', states[-1], False),
                    ('S/A pairs', pairs[-1], True),
                ]

            print(f'Episode {epoch + 1}')
            to_print = [f'{t}: {np.round(v, 2)}' for t, v, p in status if p]
            print(('{:<20}'*len(to_print)).format(*to_print))
            print()
            logging.info(','.join([str(v) for t,v,_ in status]))

        data = np.array(
            [
                timesteps,
                rewards,
                quantiles,
                flow_changes,
                flow_change_sizes,
                states,
                pairs
            ]
        )
        data = data.T

        with open('./output/trainingLogData_complete.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        if count_states:
            return count
        else:
            return None

    @classmethod
    def _make_plot(
            cls, flow, series, labels,
            ylabels=[], title=''
    ):
        if len(series) == 1:
            nrows, ncols = 1, 1
        elif len(series) == 2:
            nrows, ncols = 2, 1
        else:
            nrows, ncols = 2, 2

        style = ['r', 'g', 'black', 'orange']
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        k = 0
        for i in range(nrows):
            for j in range(ncols):
                if ncols == 1:
                    ax = axs[i]
                else:
                    ax = axs[i,j]
                ax.plot(flow, 'b--', label='flow', linewidth=.7)
                ax.set_xlabel('Time')
                ax.set_ylabel('Flow (m3/h)')
                ax2 = ax.twinx()
                ax2.plot(series[k], style[k], label=labels[k], linewidth=.7)
                ax2.grid(True)
                try:
                    ax2.set_ylabel(ylabels[k])
                except IndexError:
                    pass
                k += 1
                if k >= len(series):
                    break
            if k >= len(series):
                break

        # for ax in axs.flat:
        #     ax.label_outer()

        plt.suptitle(title)
        # fig.legend()

    @classmethod
    def plot_training_log(cls, fn):
        with open(fn, 'r') as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                data.append(list(map(float, row)))

        data = np.array(data)
        timesteps, rewards, quantiles, flow_changes, \
            flow_change_sizes, states, pairs = data.T
        fig, axs = plt.subplots(2, 2)
        x = np.arange(0,len(timesteps))
        axs[0][0].plot(np.ones(x.shape) * (365*288), 'r--', label='T', zorder=-1)
        axs[0][0].scatter(x, timesteps, label='Timesteps', c='b', s=1., zorder=1)
        # axs[0][0].legend()
        axs[0][0].grid(True)
        axs[0][0].set_ylabel('Timesteps')
        axs[0][0].set_xlabel('Episodes')

        axs[0][1].scatter(x, rewards, label='Mean reward', c='g', s=1.)
        # axs[0][1].legend()
        axs[0][1].grid(True)
        axs[0][1].set_ylabel('Mean reward')
        axs[0][1].set_xlabel('Episodes')

        axs[1][0].scatter(x, flow_changes, label='Flow changes pr timestep', c='r', s=1.)
        # axs[1][0].legend()
        axs[1][0].grid(True)
        axs[1][0].set_ylabel('Flow changes pr timestep')
        axs[1][0].set_xlabel('Episodes')

        # axs[1][1].scatter(x, quantiles, label='Quantiles', c='r', s=1.)
        # axs[1][1].set_ylabel('Difference in meters')

        axs[1][1].scatter(x, flow_change_sizes, label='Size of changes', c='b', s=1.)
        axs[1][1].set_ylabel('Size of flow change')
        # axs[1][1].plot(states, label='Sates visited', c='b', linewidth=.5)
        # axs[1][1].plot(pairs, label='Pairs visited', c='r', linewidth=.5)
        # axs[1][1].legend()
        axs[1][1].grid(True)
        axs[1][1].set_xlabel('Episodes')

        fig.suptitle('Training log for HeadModel agent')
        plt.tight_layout()
        # fig.savefig(
        #     '../report/imgs/headModelTrainingLog.png',
        #     bbox_inches='tight'
        # )

        # plt.show()
