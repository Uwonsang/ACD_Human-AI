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


if __name__ == "__main__":
    layout_path = "./overcooked_ai/overcooked_ai_py/data/layouts/big_4_fix"
    Levelplanner(layout_path)
