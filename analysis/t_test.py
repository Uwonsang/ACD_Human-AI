import pandas as pd
import scipy.stats as stats
import os
import argparse
import pickle
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('For overcooked_human_graph')
    parser.add_argument('--is_sparse', action="store_true")
    parser.add_argument('--method1', type=str, default="Ours")
    parser.add_argument('--method2', type=str, default="MAESTRO")
    parser.add_argument('--target_map', type=str, default='6054_processed')
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

    return processed_data_df



def simple_user_t_test(data, args):
    methods_mapping = {"pbt-return": "Ours", "pbt-td": "MAESTRO", "plr-td": "Robust_PLR", "random-random": "Random"}
    data["Mapped Method"] = data["Method"] + "-" + data["Category"]
    data["Mapped Method"] = data["Mapped Method"].map(methods_mapping)

    df = data[data["Map"] == args.target_map]

    values1 = df[df["Mapped Method"].str.contains(args.method1)]['Score'].values
    values2 = df[df["Mapped Method"].str.contains(args.method2)]['Score'].values

    t_stat, p_value = stats.ttest_rel(values1, values2)

    print(f"Target_map:{args.target_map}")
    print(f"T-test Results: {args.method1} vs {args.method2}")
    print({"t-statistic": t_stat, "p-value": p_value})


def run_all_user_t_tests(data):
    methods_mapping = {"pbt-return": "Ours", "pbt-td": "MAESTRO", "plr-td": "Robust_PLR", "random-random": "Random"}
    data["Mapped Method"] = data["Method"] + "-" + data["Category"]
    data["Mapped Method"] = data["Mapped Method"].map(methods_mapping)

    target_methods = ["MAESTRO", "Robust_PLR", "Random"]
    all_maps = ['6050_processed', '6051_processed', '6052_processed', '6053_processed', '6054_processed']

    for target_map in all_maps:
        df = data[data["Map"] == target_map]
        values_ours = df[df["Mapped Method"] == "Ours"]['Score'].values

        print(f"\n=== Map: {target_map} ===")
        for method in target_methods:
            values_other = df[df["Mapped Method"] == method]['Score'].values
            t_stat, p_value = stats.ttest_rel(values_ours, values_other)
            print(f"Ours vs {method}: t = {t_stat:.4f}, p = {p_value:.4f}")

if __name__ == "__main__":
    args = parse_args()
    result_dir = "Z:/overcooked_plr/overcooked_result/IEEE_ACCESS/user_study_result"
    results = load_results(result_dir, args)
    final_data = process_results(results)
    # simple_user_t_test(final_data, args)
    run_all_user_t_tests(final_data)