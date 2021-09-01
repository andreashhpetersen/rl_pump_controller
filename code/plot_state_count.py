import argparse
import numpy as np
import matplotlib.pyplot as plt


def tank_flow_bar_plot(C, Q=None):
    fig, axs = plt.subplots(5, 2, figsize=(15,15))

    state_names = [f'{i}-{i+200}' for i in range(0, 2000, 200)]

    x_ax = np.arange(16)
    x_ticks = [0] + [i for i in range(1, 16, 2)]
    x_ticklabels = ['0'] + [f'{i}' for i in range(50, 125, 10)]

    Q[C == 0] = 0
    total_visits = C.sum()
    state = 0
    for j in range(5):
        for k in range(2):
            # if state == 7:
            #     break

            T = Q[:,:,:,state,:].reshape(-1,16).mean(axis=0)
            flows = T
            # flows = T / np.abs(T.sum())

            # flows = []
            # for i in range(16):
            #     flows.append(C[:,:,:,state,i].sum())

            # total = np.round((np.sum(flows) / total_visits) * 100, 2)
            # info = f'{total}% of total visits'

            ax = axs[j][k]
            ax.bar(x_ax, flows)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)
            # ax.set_title(state_names[state] + ' - ' + info)
            ax.set_title(state_names[state])
            state += 1

    fig.suptitle('Mean Q values')
    fig.tight_layout()
    fig.savefig(
        '../report/imgs/headModelQHistogram.png',
        # bbox_inches='tight'
    )
    plt.show()

def head_vs_mean_head_bar_plot(C):
    heads, means = [], []
    for i in range(24):
        heads.append(C[:,i,:,:,:].sum())
        means.append(C[:,:,i,:,:].sum())

    steps = [(f + (7/0.16)) * 0.16 for f in range(24)]

    x_ticks = np.arange(24)
    x_ticklabels = [f'[{np.round(s, 2)}, {np.round(s + 0.16, 2)}]' for s in steps]

    plt.bar(x_ticks - 0.2, heads, width=0.4, label='Heads')
    plt.bar(x_ticks + 0.2, means, width=0.4, label='Means')
    plt.xticks(x_ticks, x_ticklabels, rotation=90)
    plt.xlabel('State')
    plt.ylabel('Number of visits')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('../report/imgs/headModelHeadVsMeanBarPlot.png')
    plt.show()

def plot_state_visits(C):
    import csv

    with open('./output/headModelTrainingLog.csv', 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(list(map(float, row)))
        data = np.array(data).T

    timesteps, rewards, quantiles,\
        flow_changes, flow_change_sizes, states, pairs = data

    c_list = C.reshape(-1)
    c_list = c_list[c_list > 0]
    c_list.sort()
    c_list = c_list[::-1]

    fig, ax = plt.subplots(1, 2, figsize=(7,3))
    ax[0].plot(pairs, c='r')
    ax[0].set_ylabel('State-/action pairs visited')
    ax[0].set_xlabel('Episodes')
    ax[0].grid(True)

    ax[1].plot(c_list, c='b')
    ax[1].set_ylabel('Number of visits')
    ax[1].set_xlabel('States ordered by number of visits')
    ax[1].grid(True)

    plt.tight_layout()

    plt.close()
    # fig.savefig('../report/imgs/headModelStatesVisited.png')

def plot_evolution_of_visits(series):
    fig, axs = plt.subplots(2,2)

    rcs = [(0,0), (0,1), (1,0), (1,1)]
    for (r, c), (t, s) in zip(rcs, series):
        ax = axs[r][c]
        ax.plot(s[:,0], c='b', label='States')
        ax.plot(s[:,1], c='r', label='Pairs')
        ax.grid(True)
        ax.legend()
        ax.title.set_text(t)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        help='path to file containing Q and C matrices'
    )
    args = parser.parse_args()


    Q, C = np.load(args.file)
    print('running in main')

    # plot_state_visits(C)
    # tank_flow_bar_plot(C, Q=Q)
    head_vs_mean_head_bar_plot(C)
