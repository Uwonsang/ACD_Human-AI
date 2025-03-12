import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np

def make_dir(save_path):
    # This code will be moved to utils.py in future
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def parse_args():
    parser = argparse.ArgumentParser('For overcooked_human_graph')
    parser.add_argument('--is_sparse', action="store_true")
    parser.add_argument('--figure_type', '-ft', default="split", type=str, choices=['split', 'full'])
    return parser.parse_args()


def load_results(result_dir, args):
    experiments = {
        'pbt': {'return': ['6050_processed', '6051_processed', '6052_processed', '6053_processed', '6054_processed'],
                'td': ['6050_processed', '6051_processed', '6052_processed', '6053_processed', '6054_processed']},
        'plr': {'td': ['6050_processed', '6051_processed', '6052_processed', '6053_processed', '6054_processed']},
        'random': {'random': ['6050_processed', '6051_processed', '6052_processed', '6053_processed', '6054_processed']}
    }

    results = {}
    person_list = os.listdir(result_dir)

    for person in person_list:
        results[person] = {}
        for strategy, sub_experiments in experiments.items():
            results[person][strategy] = {}
            for sub_strategy in sub_experiments.keys():
                results[person][strategy][sub_strategy] = {}

    for person in tqdm(person_list):
        person_path = os.path.join(result_dir, str(person))
        for strategy, sub_experiments in experiments.items():
            strategy_path = os.path.join(person_path, strategy)
            for sub_strategy, seeds in sub_experiments.items():
                sub_path = os.path.join(strategy_path, sub_strategy)
                for seed in os.listdir(sub_path):
                    seed_path = os.path.join(sub_path, seed)
                    rewards = []
                    for file_name in os.listdir(seed_path):
                        file_path = os.path.join(seed_path, file_name, seed + '.pkl')
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            epi_reward = []
                            if args.is_sparse:
                                if 'episode' not in data['info'][-1]:
                                    for info in data['info']:
                                        epi_reward.append(sum(info['sparse_r_by_agent']))
                                    rewards.append(sum(epi_reward))
                                else:
                                    rewards.append(sum(data['info'][-1]['episode']['ep_sparse_r_by_agent']))
                            else:
                                if 'episode' not in data['info'][-1]:
                                    for info in data['info']:
                                        epi_reward.append(sum(info['sparse_r_by_agent'] + info['shaped_r_by_agent']))
                                    rewards.append(sum(epi_reward))
                                else:
                                    dense_reward = sum(data['info'][-1]['episode']['ep_sparse_r_by_agent'] \
                                                   + data['info'][-1]['episode']['ep_shaped_r_by_agent'])
                                    rewards.append(dense_reward)

                        results[person][strategy][sub_strategy][seed] = rewards

    return results


def process_results(results):

    processed_data = []
    for person, methods in results.items():
        for method, categories in methods.items():
            for category, maps in categories.items():
                for map_id, values in maps.items():
                    processed_data.append([person, method, category, map_id, values[0]])

    processed_data_df = pd.DataFrame(processed_data, columns=['Person', 'Method', 'Category', 'Map', 'Score'])

    stats_df = processed_data_df.groupby(['Method', 'Category', 'Map'])['Score'].agg(['mean', 'std']).reset_index()

    return stats_df


def plot_results(final_data, args):
    methods_mapping = {"pbt-return": "Ours", "pbt-td": "MAESTRO", "plr-td": "Robust_PLR", "random-random": "Random"}
    final_data["Mapped Method"] = final_data["Method"] + "-" + final_data["Category"]
    final_data["Mapped Method"] = final_data["Mapped Method"].map(methods_mapping)

    map_mapping = {"6050_processed": r"$test_{0}$", "6051_processed": r"$test_{1}$", "6052_processed": r"$test_{2}$",
                   "6053_processed": "$test_{3}$", "6054_processed": "$test_{4}$"}
    final_data["Map"] = final_data["Map"].map(map_mapping)

    color_map = {"Ours": "skyblue", "MAESTRO": "lightcoral", "Robust_PLR": "gold", "Random": "lightgreen"}

    plt.figure(figsize=(12, 6), dpi=300)

    methods = final_data['Mapped Method'].unique()
    x_labels = sorted(final_data['Map'].unique())
    x = np.arange(len(x_labels))
    bar_width = 0.2

    for i, method in enumerate(methods):
        subset = final_data[final_data["Mapped Method"] == method]
        plt.bar(x + (i - len(methods) / 2) * bar_width, subset["mean"], yerr=subset["std"],
                capsize=5, label=method, color=color_map[method], width=bar_width)

    plt.xticks(x, x_labels)
    plt.xlabel("Unseen Layouts")
    plt.ylabel("Mean Episode Reward")
    plt.title("Performance with Real Human")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    if args.is_sparse:
        plt.savefig(os.path.join(save_dir, 'human_study_perform_sparse.pdf'), dpi=300)
    else:
        plt.savefig(os.path.join(save_dir, 'human_study_perform.pdf'), dpi=300)


def overall_figure(result_dir, save_dir, args):
    make_dir(save_dir)
    results = load_results(result_dir, args)
    final_data = process_results(results)
    plot_results(final_data, args)


if __name__ == "__main__":
    args = parse_args()
    overcooked_result_dir = "Z:/overcooked_plr/overcooked_result/scientific_reports/user_study_result"
    save_dir = './figure/human_study'
    overall_figure(overcooked_result_dir, save_dir, args)