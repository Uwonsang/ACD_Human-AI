import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse


def parse_args():
    parser = argparse.ArgumentParser('For overcooked_training_graph')
    parser.add_argument('--is_eval', action="store_true")
    parser.add_argument('--is_train', action="store_true")
    return parser.parse_args()


def smooth_data(data, alpha=0.9):
    return data.ewm(alpha=alpha, min_periods=1).mean()


def smooth_data_sma(data, window=5):
    return data.rolling(window=window, min_periods=1).mean()


def training_eval_plot(data_path, args):

    if args.is_eval:
        file_names = {
            'Ours': 'pbt_return-eval_epi_mean_reward.csv',
            'MAESTRO': 'pbt_positive-eval_epi_mean_reward.csv',
            'Robust PLR': 'plr_positive-eval_epi_mean_reward.csv',
            'Random': 'plr_random-eval_epi_mean_reward.csv'
        }
    elif args.is_train:
        file_names = {
            'Ours': 'pbt_return-train_epi_mean_reward.csv',
            'MAESTRO': 'pbt_positive-train_epi_mean_reward.csv'
        }
    else:
        file_names = {
            'Ours': 'pbt_return-train_eval_epi_mean_reward.csv',
            'MAESTRO': 'pbt_positive-train_eval_epi_mean_reward.csv',
            'Robust PLR': 'plr_positive-train_eval_epi_mean_reward.csv',
            'Random': 'plr_random-train_eval_epi_mean_reward.csv'
        }

    file_paths = {name: os.path.join(data_path, file) for name, file in file_names.items()}
    data = {name: pd.read_csv(path) for name, path in file_paths.items()}

    fig, ax = plt.subplots(figsize=(20, 10))
    for name, df in data.items():
        step_col, mean_col, min_col, max_col = df.columns
        df[mean_col] = smooth_data_sma(df[mean_col])
        sns.lineplot(data=df, x='Step', y=df[mean_col], linewidth=2.5, label=name, ax=ax)

        min_smoothed = smooth_data_sma(df[min_col])
        max_smoothed = smooth_data_sma(df[max_col])

        ax.fill_between(x=df[step_col], y1=min_smoothed, y2=max_smoothed, alpha=0.2)

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}K'.format(x * 1e-3)))

    plt.rc('font', size=15)
    plt.rc('axes', labelsize=20, titlesize=30)
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=20)
    plt.rc('figure', titlesize=20)

    plt.xlabel("Number of ego agent Updates (in K)", fontsize=30)
    plt.ylabel("Episode reward", fontsize=30)
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.show()

    if args.is_eval:
        plt.savefig('eval_graph.pdf', dpi=300)
    elif args.is_train:
        plt.savefig('train_graph.pdf', dpi=300)
    else:
        plt.savefig('train_eval_graph.pdf', dpi=300)


if __name__ == "__main__":
    all_args = parse_args()
    data_path = './data'
    training_eval_plot(data_path, all_args)
