from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, BASE_REW_SHAPING_PARAMS
import os
import shutil
import numpy as np
from itertools import combinations
import ast
from tqdm import tqdm

def Levelplanner(layout_path):

    layout_list = [filename for filename in os.listdir(layout_path) if filename.endswith('.layout')]
    processed_list = [filename.replace('.layout', '') for filename in layout_list]
    error_list = []

    for i in range(len(processed_list)):
        mdp_params = {'layout_name': processed_list[i], 'start_order_list': None}
        mdp_params.update({
            "rew_shaping_params": BASE_REW_SHAPING_PARAMS,
            "layouts_dir": layout_path
        })

        mdp_fn = lambda: OvercookedGridworld.from_layout_name(**mdp_params)
        base_mdp = mdp_fn()

        try:
            mlp = MediumLevelPlanner.from_pickle_or_compute(
                mdp=base_mdp,
                mlp_params=NO_COUNTERS_PARAMS,
                force_compute=False)
        except Exception as e:
            print(f"Error with layout: {processed_list[i]}, Error: {e}")
            error_list.append(processed_list[i])

    print("Layouts with errors:", error_list)


def change_layout_info(layout_path, error_list):
    metric_path = layout_path + "/metric/normalized_metric.npy"
    # sim_metric = np.load(metric_path)

    error_mapping = {
        '2346_processed': '6045_processed',
        '3256_processed': '6046_processed',
        '3263_processed': '6047_processed',
        '3493_processed': '6048_processed',
        '35_processed': '6049_processed'}

    layout_list = [filename for filename in os.listdir(layout_path) if filename.endswith('.layout')]

    for error_name in error_list:

        new_name = error_mapping[error_name]
        old_file_path = os.path.join(layout_path, f"{error_name}.layout")
        new_file_path = os.path.join(layout_path, f"{new_name}.layout")

        shutil.copyfile(new_file_path, old_file_path)
        print(f"Overwritten: {new_file_path} → {old_file_path}")

        os.remove(new_file_path)
        print(f"Deleted: {new_file_path}")


def build_hamming_list(layout_path):
    num_train_layout = 6000
    layout_list = sorted([filename for filename in os.listdir(layout_path) if filename.endswith('.layout')], key=lambda x: int(x.split('_')[0]))
    all_layouts = []

    for file_name in tqdm(layout_list):
        file_path = os.path.join(layout_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            layout_dict = ast.literal_eval(file_content)
            grid = [layout_row.strip() for layout_row in layout_dict["grid"].split('\n')]
            layout_grid = [[c for c in row] for row in grid]
            all_layouts.append(layout_grid)

    value_array = np.zeros((num_train_layout, num_train_layout), dtype=int)
    distance_list = []

    num_combinations = (num_train_layout * (num_train_layout - 1)) // 2
    for pair in tqdm(combinations(range(num_train_layout), 2), total=num_combinations):
        # distance = hamming_distance(all_layouts[pair[0]], all_layouts[pair[1]])
        distance = hamming_eqaul_distance(all_layouts[pair[0]], all_layouts[pair[1]])
        distance_list.append(distance)
        value_array[pair[0]][pair[1]] = distance

    min_val = np.min(distance_list)
    max_val = np.max(distance_list)

    normalized_array = (value_array - min_val) / (max_val - min_val)
    normalized_array[value_array == 0] = 0

    transpose_array = normalized_array.transpose()
    result_array = normalized_array + transpose_array

    # save_path = os.path.join(layout_path, "metric/normalized_metric.npy")
    save_path = os.path.join(layout_path, "metric/normalized_equal_metric.npy")

    np.save(save_path, result_array)


def hamming_distance(individual1, individual2):
    individual1 = np.array(individual1)
    individual2 = np.array(individual2)

    individual1 = np.where((individual1 == '1') | (individual1 == '2'), ' ', individual1)
    individual2 = np.where((individual2 == '1') | (individual2 == '2'), ' ', individual2)

    distance = np.sum(individual1 != individual2)

    return distance


def hamming_eqaul_distance(individual1, individual2):
    individual1 = np.array(individual1)
    individual2 = np.array(individual2)

    individual1 = np.where((individual1 == '1') | (individual1 == '2'), ' ', individual1)
    individual2 = np.where((individual2 == '1') | (individual2 == '2'), ' ', individual2)

    distance = np.sum(individual1 == individual2)

    return distance

if __name__ == "__main__":
    layout_path = "/data/overcooked_layout/big_4_fix"
    Levelplanner(layout_path)
    # layout_path = "/data/overcooked_layout/big_4_fix"
    # error_list = ['2346_processed', '3256_processed', '3263_processed', '3493_processed', '35_processed']
    # change_layout_info(layout_path, error_list)
    # build_hamming_list(layout_path)
