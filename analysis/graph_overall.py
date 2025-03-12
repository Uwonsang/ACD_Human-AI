import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def make_dir(save_path):
    # This code will be moved to utils.py in future
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def parse_args():
    parser = argparse.ArgumentParser('overcooked_graph')
    parser.add_argument('--is_sparse', action="store_true")
    parser.add_argument('--figure_type', '-ft', default="full", type=str, choices=['split', 'full'])
    return parser.parse_args()


def read_and_process_csv_full(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(df.index[-1])
    df['seed_mean'] = df.iloc[:, 1:].mean(axis=1)

    return df

def read_and_process_csv_split(file_path):
    df = pd.read_csv(file_path).T
    df.columns = df.iloc[0]  # 첫 번째 행을 열 이름으로 사용
    df = df.drop(index=df.index[0]).reset_index(drop=True)
    df['std'] = df.iloc[:, :-1].std(axis=1)  # 표준편차 계산
    return df


def overall_figure(result_dir, save_dir, args):
    make_dir(save_dir)

    if args.is_sparse:
        file_names = {
            'Ours': 'scientific_reports_3.pbt_return vs Proxy_sparse.csv',
            'MAESTRO': 'scientific_reports_3.pbt_td vs Proxy_sparse.csv',
            'Robust PLR': 'scientific_reports_2.plr_td vs Proxy_sparse.csv',
            'Random': 'scientific_reports_1.random_random vs Proxy_sparse.csv'
        }
    else:
        file_names = {
            'Ours': 'scientific_reports_3.pbt_return vs Proxy.csv',
            'MAESTRO': 'scientific_reports_3.pbt_td vs Proxy.csv',
            'Robust PLR': 'scientific_reports_2.plr_td vs Proxy.csv',
            'Random': 'scientific_reports_1.random_random vs Proxy.csv'
        }

    file_paths = {name: os.path.join(result_dir, file) for name, file in file_names.items()}


    if args.figure_type == "split":
        data = {name: read_and_process_csv_split(path) for name, path in file_paths.items()}
        layout_name = [f'$test_{{{i}}}$' for i in range(5)]
        bar_width = 0.2
        index = np.arange(len(layout_name))
        plt.figure(figsize=(12, 6), dpi=300)

        colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen']
        for i, (name, df) in enumerate(data.items()):
            plt.bar(index + i * bar_width, df['mean'], bar_width, label=name, color=colors[i], yerr=df['std'], capsize=10)

        plt.xlabel('Unseen Layouts')
        plt.ylabel('Mean Episode Reward')
        plt.title('Performance with human proxy agent')
        plt.xticks(index + 1.5 * bar_width, layout_name, fontstyle='italic')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        if args.is_sparse:
            plt.savefig(os.path.join(save_dir, 'overall_split_sparse.pdf'), dpi=300)
        else:
            plt.savefig(os.path.join(save_dir, 'overall_split.pdf'), dpi=300)
        # plt.show()

    elif args.figure_type == "full":
        data = {name: read_and_process_csv_full(path) for name, path in file_paths.items()}
        layout_name = list(file_names.keys())

        index = np.arange(len(layout_name))
        bar_width = 0.9
        plt.figure(figsize=(12, 6), dpi=300)

        colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen']
        for i, (name, df) in enumerate(data.items()):
            plt.bar(index[i], df['seed_mean'].mean(), width=bar_width, label=name,
                    color=colors[i], yerr=df['seed_mean'].std(), capsize=10)

        # plt.gca().axes.xaxis.set_visible(False)
        plt.xlabel('Unseen Layouts')
        plt.ylabel('Mean Episode Reward')
        plt.title('Performance with human proxy agent')
        plt.xticks(index, layout_name, fontstyle='italic')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        if args.is_sparse:
            plt.savefig(os.path.join(save_dir, 'overall_full_sparse.pdf'), dpi=300)
        else:
            plt.savefig(os.path.join(save_dir, 'overall_full.pdf'), dpi=300)
        # plt.show()


if __name__ == "__main__":
    args = parse_args()
    overcooked_result_dir = "/data/overcooked_plr/overcooked_result/scientific_reports/result/Ego_Proxy"
    save_dir = '/app/analysis/figure/overall/'
    overall_figure(overcooked_result_dir, save_dir, args)
