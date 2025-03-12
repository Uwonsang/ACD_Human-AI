import sys
import os

import time
import numpy as np
import torch
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter

local_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if local_path not in sys.path:
    sys.path.insert(0, local_path)

from test import evaluate
from level_replay.arguments import parser
from level_replay import utils
from level_replay.model import OvercookedPolicy
from level_replay.envs import make_lr_venv
from overcooked_ai.overcooked_ai_py.data.layouts import read_layout_dict


def eval_train_layout(args, model, impossible_layouts, log_dir, device):
    train_layout_start_idx = 0
    
    result_list = []
    sparse_result_list = []
    total_iterations = (args.num_train_seeds - train_layout_start_idx + args.num_processes_test - 1) // args.num_processes_test

    for _ in tqdm(range(total_iterations), desc="Evaluating layouts"):
        print(f"Evalute layout {train_layout_start_idx} ~ {train_layout_start_idx+args.num_processes_test-1}")
        eval_envs, _ = make_lr_venv(
            num_envs=args.num_processes_test,
            seeds=None, device=device,
            all_args=args,
            num_levels=0, start_level=0,
            no_ret_normalization=args.no_ret_normalization,
            level_sampler=None,
            human_proxy_num=1,
            start_seed=train_layout_start_idx,
            obp_eval_map=False,
            activate_planner=True,
            )
        train_layout_start_idx += args.num_processes_test
        overcooked = 'env_name' in eval_envs.__dict__.keys() and eval_envs.env_name == "Overcooked-v0"

        eval_episode_rewards, _, _ , eval_episode_sparse_reward = evaluate(args, model, args.num_test_seeds, device, eval_envs=eval_envs, num_processes=args.num_processes_test, activate_planner=True)

        result_list.append(eval_episode_rewards)
        sparse_result_list.append(eval_episode_sparse_reward)

        with open(os.path.join(log_dir, "train_6000_result.pkl"), 'wb') as file1:
            pickle.dump(result_list, file1)

        with open(os.path.join(log_dir, "train_6000_sparse_result.pkl"), 'wb') as file2:
            pickle.dump(sparse_result_list, file2)

        
        eval_envs.close()
        
    return result_list, sparse_result_list

def check_impossible_layout(args):
    def is_linear_path(grid):
        rows = len(grid)
        cols = len(grid[0])

        # Directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # Check if the cell is within the grid and is passable
        def is_passable(x, y):
            return 0 <= x < rows and 0 <= y < cols and grid[x][y] in '12 '

        # Count the number of passable neighbors
        def count_passable_neighbors(x, y):
            return sum(is_passable(x + dx, y + dy) for dx, dy in directions)

        # Find the starting point and count the number of branching points
        start_points = 0
        for i in range(rows):
            for j in range(cols):
                if is_passable(i, j):
                    neighbors = count_passable_neighbors(i, j)
                    # A start/end point has exactly one passable neighbor
                    if neighbors == 1:
                        start_points += 1
                    # A branching point or a crossing would have more than 2 neighbors
                    elif neighbors > 2:
                        return False

        # The grid is linear if there are exactly two start/end points
        return start_points == 2

    layout_dir = os.path.join(args.layouts_dir, args.layouts_type)
    impossible_layout = []
    impossible_cnt = 0
    for layout_num in range(args.num_train_seeds):
        layout_name = f"{layout_num}_processed"
        base_layout_params = read_layout_dict(layout_name, layout_dir)
        grid_str = base_layout_params['grid']
        grid = [line.strip() for line in grid_str.split('\n')]
        
        
        
        if is_linear_path(grid):
            impossible_layout.append(layout_num)
            impossible_cnt+=1
            print(f"{layout_num} is not possible for human proxy")

    print("============")
    print(f"{args.num_train_seeds-impossible_cnt} layouts can make human proxy")
    print(f"{impossible_cnt} layouts can't make human proxy")
    
    return impossible_layout
            
        
def analyse_pickle(pickle_path):
    with open(os.path.join(pickle_path, "train_6000_result.pkl"), "rb") as fr:
        data = pickle.load(fr)
        
    score_dict = {}
    for i, score in enumerate(data):
        for j in range(len(score)):
            idx = i*len(score)+j
            score_dict[idx] = score[j]
    
    
    sorted_score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]))
    
    with open(os.path.join(pickle_path, "train_6000_sorted_score.pkl"), 'wb') as file1:
        pickle.dump(sorted_score_dict, file1)
            
            
    with open(os.path.join(pickle_path, "train_6000_sparse_result.pkl"), "rb") as fr:
        sparse_data = pickle.load(fr)
        
    sparse_score_dict = {}
    for i, sparse_score in enumerate(sparse_data):
        for j in range(len(sparse_score)):
            idx = i*len(sparse_score)+j
            sparse_score_dict[idx] = sparse_score[j]
            
    sorted_sparse_score_dict = dict(sorted(sparse_score_dict.items(), key=lambda x: x[1]))
    with open(os.path.join(pickle_path,"train_6000_sorted_sparse_score.pkl"), 'wb') as file2:
        pickle.dump(sorted_sparse_score_dict, file2)

    return score_dict, sparse_score_dict

def get_score_dict(pickle_dir): 
    with open(os.path.join(pickle_dir, "train_6000_sorted_score.pkl"), 'rb')as fr:
        score_dict = pickle.load(fr)
        
    with open(os.path.join(pickle_dir, "train_6000_sorted_sparse_score.pkl"), 'rb')as fr:
        sparse_score_dict = pickle.load(fr)
        
    return score_dict, sparse_score_dict
        
        
    
def get_difficulty(score_dict, log_dir):
    score_list = list(score_dict.values())
    
    mean_reward = np.mean(score_list)
    std_reward = np.std(score_list)
    median_reward = np.median(score_list)
    
    def calculate_difficulty(score):
        if score > mean_reward + 1.5 * std_reward:
            return 0  # Very Easy
        elif score > mean_reward + 0.5 * std_reward:
            return 1  # Easy
        elif score > mean_reward - 0.5 * std_reward:
            return 2  # Medium
        elif score > mean_reward - 1.5 * std_reward:
            return 3  # Hard
        else:
            return 4  # Very Hard
        
    def visualize_diff():
        scores_list = list(score_dict.values())
        rounded_scores = [round(score) for score in scores_list]

        score_frequencies = Counter(rounded_scores)

        scores, frequencies = zip(*sorted(score_frequencies.items()))
        score_freq_dict = {score: freq for score, freq in zip(scores, frequencies)}
        
        colors = ['#6AA84F', '#B5E61D', '#FFD966', '#F6B26B', '#E06666']

        difficulties = np.array([calculate_difficulty(score) for score in scores])
        color_map = [colors[difficulty] for difficulty in difficulties]

        rc_param = {
            'axes.labelsize': 18.5,
            'axes.titlesize': 23,
            'legend.fontsize':12
            }
        plt.rcParams.update(rc_param)
        
        plt.figure(figsize=(12, 6))

        for score, color in zip(scores, color_map):
            plt.bar(score, score_freq_dict[score], color=color)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(colors[::-1], ['Very Hard', 'Hard', 'Medium', 'Easy', 'Very Easy'])]
        plt.legend(handles=legend_handles, title="Difficulty", loc="best")
        plt.savefig(os.path.join(log_dir,  f"difficulty.png"), dpi=300)
        plt.savefig(os.path.join(log_dir,  f"difficulty.pdf"))
        

    difficult_dict = {}
    diff_cnt = [0, 0, 0, 0, 0]
    for map_name, reward in score_dict.items():
        difficulty = calculate_difficulty(reward)
        difficult_dict[map_name] = difficulty
        diff_cnt[difficulty] += 1
        
    visualize_diff()
    with open(os.path.join(log_dir, "difficulty_dict.pkl"), 'wb') as file1:
        pickle.dump(difficult_dict, file1)
        
    
    return difficult_dict, diff_cnt
    
    
def get_co_player_buffer(checkpoint):
    with open(checkpoint, "rb") as fr:
        co_player_buffer_data = pickle.load(fr)    

    buffer_list = co_player_buffer_data["buffer"]
    co_player_list = co_player_buffer_data["co_player"]
    total_update_list = co_player_buffer_data["total_update"]
    ppo_update_list = co_player_buffer_data["ppo_update"]
    
    
    co_player_buffer_dict = dict()
    ppo_update_idx_dict = dict()
    idx_cnt = -1
    for (update_step, ppo_update, buffer, co_player) in zip(total_update_list, ppo_update_list, buffer_list, co_player_list):
        if ppo_update not in ppo_update_idx_dict:
            ppo_update_idx_dict[ppo_update] = idx_cnt
            idx_cnt += 1
        
        if co_player not in co_player_buffer_dict:
            co_player_buffer_dict[co_player] = dict()
            
        co_player_buffer_dict[co_player][ppo_update] = buffer
    
    return co_player_buffer_dict



# Checks the key value in a buffer, gets the corresponding difficulty, and stores it   
def get_difficulty_from_buffer(co_player_buffer_dict, layout_difficulty_dict, log_dir, num_graphs=5):
    buffer_max_size = 1000
    buffer_difficulty_dict = dict()
    
    for co_player, step_buffer_dict in co_player_buffer_dict.items():
        if co_player not in buffer_difficulty_dict:
            buffer_difficulty_dict[co_player] = dict()
        buffer_skip_cnt = int(len(step_buffer_dict) /num_graphs)      
        for idx, (ppo_num, buffer) in enumerate(step_buffer_dict.items()):
            if buffer_skip_cnt == 0:
                buffer_skip_cnt = 1
            if idx % buffer_skip_cnt != 0 or len(buffer_difficulty_dict[co_player])>=num_graphs:
                continue
            diff_cnt = [0, 0, 0, 0, 0]
            for layout, value in buffer.items():
                difficulty = layout_difficulty_dict[layout]
                diff_cnt[difficulty] += 1
            for i in range(5):
                diff_cnt[i] = diff_cnt[i] / buffer_max_size * 100
            buffer_difficulty_dict[co_player][ppo_num] = diff_cnt
    

    
    with open(os.path.join(log_dir, "step_buffer_difficulty.pkl"), 'wb') as file1:
            pickle.dump(buffer_difficulty_dict, file1)
    return buffer_difficulty_dict
    
    
 
#Visualize difficulty_buffer
#Visualize in the form of a stacked bar graph
#Visualize in the form of a stacked bar graph Normalized by the number of layouts in each difficulty level
def visualize_buffer_diff(buffer_difficulty_dict, diff_cnt, log_dir, graph_name=""):    
    color_map = {
       'Very Easy': '#6AA84F',
        'Easy': '#B5E61D',
        'Medium': '#FFD966',
        'Hard': '#F6B26B',
        'Very Hard': '#E06666'
    }
    graph_path = os.path.join(log_dir, graph_name)
    utils.make_dir(graph_path)
    for co_player, step_buffer_dict in buffer_difficulty_dict.items():
        ppo_step = list(step_buffer_dict.keys())
        step = list(range(len(ppo_step)))
        difficulty = np.array(list(step_buffer_dict.values()))
        plt.figure(figsize=(len(step)+3, 5))
    
        plt.bar(step, difficulty[:, 0], label="Very Easy", color=color_map['Very Easy'])
        plt.bar(step, difficulty[:, 1], bottom=difficulty[:, 0], label="Easy", color=color_map['Easy'])
        plt.bar(step, difficulty[:, 2], bottom=difficulty[:, 0]+difficulty[:, 1], label="Medium", color=color_map['Medium'])
        plt.bar(step, difficulty[:, 3], bottom=difficulty[:, 0]+difficulty[:, 1]+difficulty[:, 2], label="Hard", color=color_map['Hard'])
        plt.bar(step, difficulty[:, 4], bottom=difficulty[:, 0]+difficulty[:, 1]+difficulty[:, 2]+difficulty[:, 3], label="Very Hard", color=color_map['Very Hard'])
        
        plt.xticks(step, ppo_step) 
        plt.ylabel('Percent', fontsize=14)
        plt.xlabel('PPO update', fontsize=14)
        plt.title(f'{graph_name}', fontsize=18)
        
        plt.tight_layout()
        # save
        plt.savefig(os.path.join(graph_path, f"{co_player}_difficulty.png"), dpi=300)
        plt.savefig(os.path.join(graph_path, f"{co_player}_difficulty.pdf"))
        plt.close()


    
    
def visualize_buffer_pie(buffer_difficulty_dict, log_dir, graph_name="step_buffer"):
    graph_path = os.path.join(log_dir, graph_name)
    corrected_difficulty_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    color_map = {
       'Very Easy': '#6AA84F',
        'Easy': '#B5E61D',
        'Medium': '#FFD966',
        'Hard': '#F6B26B',
        'Very Hard': '#E06666'
    }

    # Sum the values for each difficulty key across all steps
    for step_buffer in buffer_difficulty_dict.values():
        for step_difficulties in step_buffer.values():
            for difficulty, count in enumerate(step_difficulties):
                corrected_difficulty_counts[difficulty] += count

    # Prepare data for the corrected pie chart
    corrected_sizes = list(corrected_difficulty_counts.values())
    
    difficulty_labels_with_text = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
    
    rc_param = {'axes.titlesize': 23}
    plt.rcParams.update(rc_param)
    
    # Create the pie chart with legends
    plt.figure(figsize=(6, 6))
    plt.pie(corrected_sizes, labels=difficulty_labels_with_text, autopct='%1.1f%%', startangle=0, 
        colors=color_map.values(), counterclock=False, textprops={'fontsize':15})  # Sorted and clockwise
    plt.axis('equal')  # Ensure pie is drawn as a circle.
    
    plt.title(f'{graph_name}', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_path,  f"difficulty_{graph_name}_pie.png"), dpi=300)
    plt.savefig(os.path.join(graph_path,  f"difficulty_{graph_name}_pie.pdf"))


def eval_train_layout_with_proxy(args, log_dir):
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
        
    args.cuda =  torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')

    overcooked_result_dir = "/app/overcooked_result"
    args.xpid = "lr-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), "6000_level")    
    
    impossible_layouts = check_impossible_layout(args=args)
    
    tmp_env, _ = make_lr_venv(
            num_envs=1, 
            seeds=None, device=device,
            all_args=args,
            num_levels=0, start_level=0,
            no_ret_normalization=args.no_ret_normalization,
            level_sampler=None,
            activate_planner=True,
            obp_eval_map=False,
            human_proxy_num=1)

    model = OvercookedPolicy(tmp_env.observation_space.shape, tmp_env.action_space.n, args).to(device)

    tmp_env.close()

    checkpoint = args.checkpoint_path

    model_checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    
    eval_train_layout(args=args,model=model, impossible_layouts=impossible_layouts, log_dir=log_dir, device=device)
    score_dict, sparse_score_dict = analyse_pickle(log_dir)
    
    return score_dict, sparse_score_dict
     
        
if __name__ == "__main__":
    args = parser.parse_args()
    xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    
    log_dir = None 

    log_dir = "/app/overcooked_result_log/diff_score_graph/ours/lr-20250225-113853"
    if log_dir is None:   
        log_dir = os.path.expandvars(os.path.expanduser("%s/%s/%s/%s" % (args.result_log_dir, "diff_score_graph", args.model_name, xpid)))
        utils.make_dir(log_dir)

        eval_train_layout_with_proxy(args, log_dir) 
    
    score_dict, sparse_score_dict = get_score_dict(log_dir)
    layout_difficulty_dict, diff_cnt  = get_difficulty(score_dict ,log_dir)
    
    methods = ['return', 'td']
    graph_names = ['Ours', 'MAESTRO']
    for method, graph_name in zip(methods, graph_names):
        checkpoint = f"/data/overcooked_plr/overcooked_result/scientific_reports/3.pbt/{method}/seed2/co_player_buffer/co_player_buffer.pkl"
    
        co_player_buffer_dict = get_co_player_buffer(checkpoint)
        buffer_difficulty_dict_5len= get_difficulty_from_buffer(co_player_buffer_dict, layout_difficulty_dict, log_dir, num_graphs=5)
        buffer_difficulty_dict_all= get_difficulty_from_buffer(co_player_buffer_dict, layout_difficulty_dict, log_dir, num_graphs=1)

        visualize_buffer_diff(buffer_difficulty_dict_5len, diff_cnt, log_dir, graph_name=graph_name)
        visualize_buffer_pie(buffer_difficulty_dict_all, log_dir, graph_name=graph_name)
