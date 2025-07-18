import scipy.stats as stats
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


def overall_figure(result_dir, args):

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
    data = {name: read_and_process_csv_split(path) for name, path in file_paths.items()}

    return data


def run_t_tests_on_dict(data_dict):
    baseline = "Ours"
    targets = [k for k in data_dict if k != baseline]
    ours_df = data_dict[baseline].drop(columns=['mean', 'std'], errors='ignore')

    layouts = sorted(ours_df.index.tolist())

    print("=== Paired t-tests (per layout): Ours vs each method ===\n")
    for layout in layouts:
        print(f"\n--- Layout: {layout} ---")
        ours_vals = ours_df.loc[layout].astype(float).values

        for target in targets:
            target_df = data_dict[target].drop(columns=['mean', 'std'], errors='ignore')
            target_vals = target_df.loc[layout].astype(float).values

            t_stat, p_val = stats.ttest_rel(ours_vals, target_vals)
            print(f"Ours vs {target}: t = {t_stat:.4f}, p = {p_val:.4f}")



if __name__ == "__main__":
    args = parse_args()
    overcooked_result_dir = "Z:/overcooked_plr/overcooked_result/IEEE_ACCESS/result/Ego_Proxy"
    data = overall_figure(overcooked_result_dir, args)
    run_t_tests_on_dict(data)
