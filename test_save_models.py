import os
from level_replay.arguments import parser
from test import evaluate, evaluate_e3t
from level_replay import utils

import time
import numpy as np
import torch
from tqdm import tqdm

from level_replay.model import OvercookedPolicy_E3T, OvercookedPolicy
from level_replay.envs import make_lr_venv
import pandas as pd
from level_replay.utils import get_target_list


def make_eval_env(args, test_method, visualize=False):

    args.use_render = visualize
    if test_method == 1:  # ego vs human proxy
        eval_envs, _ = make_lr_venv(
            num_envs=args.num_processes_test,
            seeds=None, device=device,
            all_args=args,
            no_ret_normalization=args.no_ret_normalization,
            level_sampler=None,
            activate_planner=True,
            obp_eval_map=True,
            human_proxy_num=1)
        
    elif test_method == 4:  # human proxy vs human proxy
        eval_envs, _ = make_lr_venv(
            num_envs=args.num_processes_test,
            seeds=None, device=device,
            all_args=args,
            no_ret_normalization=args.no_ret_normalization,
            level_sampler=None,
            activate_planner=True,
            obp_eval_map=True,
            human_proxy_num=2)
        
    else:  # ego vs ego  or ego vs co_player
        eval_envs, _ = make_lr_venv(
            num_envs=args.num_processes_test,
            seeds=None, device=device,
            all_args=args,
            no_ret_normalization=args.no_ret_normalization,
            level_sampler=None,
            activate_planner=False,
            obp_eval_map=True,
            human_proxy_num=0)

    return eval_envs


def select_test_method():
    method = {
        1: " Ego_Proxy",
        2: " Ego_Ego",
        3: " Ego_Co_player",
        4: " Proxy_Proxy",
    }
    print("<Select Test Metod>")
    print("1: Ego vs Human Proxy")
    print("2: Ego vs Ego")
    print("3: Ego vs Co_player")
    print("4: Human Proxy vs Human Proxy")
    test_method = int(input("select num: "))
    assert 1 <= test_method <= 4, "Pleas select correct method"
    return test_method, method[test_method]


def save_print_result(args, result_list, result_var_list, sparse_result_list, sparse_result_var_list, repeat_num, test_name):
    log_dir = os.path.expandvars(os.path.expanduser("%s/%s" % (args.result_log_dir, args.xpid)))
    utils.make_dir(log_dir)
    
    df = pd.DataFrame(result_list, columns=['env1', 'env2', 'env3', 'env4', 'env5'])
    df.loc['mean'] = df.mean()
    df.to_csv(os.path.join(log_dir, f"{test_name}.csv"))
    
    df_sparse = pd.DataFrame(sparse_result_list, columns=['env1', 'env2', 'env3', 'env4', 'env5'])
    df_sparse.loc['mean'] = df_sparse.mean()
    df_sparse.to_csv(os.path.join(log_dir, f"{test_name}_sparse.csv"))
    
    df_var = pd.DataFrame(result_var_list, columns=['env1', 'env2', 'env3', 'env4', 'env5'])
    df_var.to_csv(os.path.join(log_dir, f"{test_name}_var.csv"))
     
    df_sparse_var = pd.DataFrame(sparse_result_var_list, columns=['env1', 'env2', 'env3', 'env4', 'env5'])
    df_sparse_var.to_csv(os.path.join(log_dir, f"{test_name}_sparse_var.csv"))
                
    print(" ")
    result_list = df.iloc[:-1].values.tolist()  # 마지막 행(평균) 제외
    mean_results = df.loc['mean'].values.tolist()
    print(f"<{test_name} Mean Rewards for {repeat_num} times>")    
    for seed_idx in range(df.shape[0]-1):
        print(f"\t[seed {seed_idx+1}]", end=" ")
        for env_idx in range(5):
            print(f"env{env_idx+1}: {result_list[seed_idx][env_idx]:.3f},", end="\t")
        print(" ")
    print("\t[Mean]", end=" ")
    for env_idx, mean_result in enumerate(mean_results):
        print(f"env{env_idx+1}: {mean_result:.3f},", end="\t")
    print("\n")
    
    print(" ")
    sparse_mean_results = df_sparse.loc['mean'].values.tolist()
    print(f"<{test_name} Mean Sparse Rewards for {repeat_num} times>")    
    for seed_idx in range(df_sparse.shape[0]-1):
        print(f"\t[seed {seed_idx+1}]", end=" ")
        for env_idx in range(5):
            print(f"env{env_idx+1}: {sparse_result_list[seed_idx][env_idx]:.3f},", end="\t")
        print(" ")
    print("\t[Mean]", end=" ")
    for env_idx, sparse_mean_result in enumerate(sparse_mean_results):
        print(f"env{env_idx+1}: {sparse_mean_result:.3f},", end="\t")
    print(" ")


def eval_ego_proxy(args, model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t):

    while True:
        print("\n Ego vs Proxy")
        print("\n<Ego Agent List>")
        for idx, checkpoint in enumerate(checkpoint_list):
            print(f"[{idx+1}]: {checkpoint[0]}, {checkpoint[1]} seeds")
        
        x = int(input("select num for evaluate. select 0 to close: "))
        if x < 1 or x > len(checkpoint_list):
            print("exit")
            break
        
        result_list = []
        result_var_list = []
        sparse_result_list = []
        sparse_result_var_list = []
        test_name = f"{checkpoint_list[x-1][0]} vs Proxy" 
        print(f"\n{test_name}\n") 
        for seed_idx in range(checkpoint_list[x-1][1]):
            print(f"seed {seed_idx+1}")
            if torch.cuda.is_available():
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = 'cpu'
            model_checkpoint = torch.load(checkpoint_list[x-1][2][seed_idx], map_location=map_location)
            model.load_state_dict(model_checkpoint["model_state_dict"])

            seed_result_list = []
            sparse_seed_result_list = []
            for i in tqdm(range(repeat_num)):
                if is_e3t:
                    eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate_e3t(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs, num_processes=args.num_processes_test, use_render=False, store_traj=False)
                else:
                    eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs, num_processes=args.num_processes_test, use_render=False, store_traj=False)
                seed_result_list.append(eval_episode_rewards)
                sparse_seed_result_list.append(eval_episode_sparse_reward)
            seed_result_list = np.array(seed_result_list)
            sparse_seed_result_list = np.array(sparse_seed_result_list)

            result_list.append(seed_result_list.mean(axis=0))
            result_var_list.append(seed_result_list.var(axis=0))
            sparse_result_list.append(sparse_seed_result_list.mean(axis=0))
            sparse_result_var_list.append(sparse_seed_result_list.var(axis=0))
            
            if visualize:
                run_dir = os.path.expandvars(os.path.expanduser("%s/%s/%s/%s" % (args.result_log_dir, args.xpid, test_name, f"seed{seed_idx}")))
                utils.make_dir(run_dir)
                args.run_dir = run_dir
                visualize_env = make_eval_env(args, 1, visualize=True)
                evaluate(args, model, args.final_num_test_seeds, 
                         device, use_render=True, store_traj=True, isFinalEval=True, eval_envs=visualize_env)

        save_print_result(args, result_list, result_var_list, sparse_result_list, sparse_result_var_list, repeat_num, test_name)


def eval_ego_ego(args, model, co_model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t):
    while True:
        print("\n Ego vs Ego")
        print("\n<Ego Agent List>")
        for idx, checkpoint in enumerate(checkpoint_list):
            print(f"[{idx+1}]: {checkpoint[0]}, {checkpoint[1]} seeds")
        
        x = int(input("select num for evaluate. select 0 to close: "))
        if x < 1 or x > len(checkpoint_list):
            print("exit")
            break
        
        result_list = []
        result_var_list = []
        sparse_result_list = []
        sparse_result_var_list = []
        
        test_name = f"{checkpoint_list[x-1][0]} vs {checkpoint_list[x-1][0]}"
        print(f"\n{test_name}\n")
        
        for seed_idx in range(checkpoint_list[x-1][1]):
            print(f"seed {seed_idx+1}")
            if torch.cuda.is_available():
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = 'cpu'
            model_checkpoint = torch.load(checkpoint_list[x-1][2][seed_idx], map_location=map_location)
            model.load_state_dict(model_checkpoint["model_state_dict"])
            co_model.load_state_dict(model_checkpoint["model_state_dict"])
            
            seed_result_list = []
            sparse_seed_result_list = []
            for i in tqdm(range(repeat_num)):
                if is_e3t:
                    eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate_e3t(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs, num_processes=args.num_processes_test, use_render=False, store_traj=False, co_player=co_model)
                else:
                    eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs, co_player=co_model)
                seed_result_list.append(eval_episode_rewards)
                sparse_seed_result_list.append(eval_episode_sparse_reward)
            seed_result_list = np.array(seed_result_list)
            sparse_seed_result_list = np.array(sparse_seed_result_list) 
            result_list.append(seed_result_list.mean(axis=0))
            result_var_list.append(seed_result_list.var(axis=0))
            sparse_result_list.append(sparse_seed_result_list.mean(axis=0))
            sparse_result_var_list.append(sparse_seed_result_list.var(axis=0))
            if visualize:
                run_dir = os.path.expandvars(os.path.expanduser("%s/%s/%s/%s" % (args.result_log_dir, args.xpid, test_name, f"seed{seed_idx}")))
                utils.make_dir(run_dir)
                args.run_dir = run_dir
                visualize_env = make_eval_env(args, 2, visualize=True)
                evaluate(args, model, args.final_num_test_seeds, 
                         device, use_render=True, store_traj=True, isFinalEval=True, eval_envs=visualize_env)

        save_print_result(args, result_list, result_var_list, sparse_result_list, sparse_result_var_list, repeat_num, test_name)


def eval_proxy_proxy(args, model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t):
    x = 1
    
    result_list = []
    result_var_list = []
    sparse_result_list = []
    sparse_result_var_list = []
    test_name = "Proxy vs Proxy"
    
    print("\nProxy vs Proxy")
    for seed_idx in range(checkpoint_list[x-1][1]):
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        model_checkpoint = torch.load(checkpoint_list[x-1][2][seed_idx], map_location=map_location)
        model.load_state_dict(model_checkpoint["model_state_dict"])
        
        seed_result_list = []
        sparse_seed_result_list = []
        for i in tqdm(range(repeat_num)):
            if is_e3t:
                eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate_e3t(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs, num_processes=args.num_processes_test, use_render=False, store_traj=False)
            else:
                eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs)
                
            eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs)
            seed_result_list.append(eval_episode_rewards)
            sparse_seed_result_list.append(eval_episode_sparse_reward)
        seed_result_list = np.array(seed_result_list)
        sparse_seed_result_list = np.array(sparse_seed_result_list)
        result_list.append(seed_result_list.mean(axis=0))
        result_var_list.append(seed_result_list.var(axis=0))
        sparse_result_list.append(sparse_seed_result_list.mean(axis=0))
        sparse_result_var_list.append(sparse_seed_result_list.var(axis=0))
        if visualize:
            run_dir = os.path.expandvars(os.path.expanduser("%s/%s/%s/%s" % (args.result_log_dir, args.xpid, test_name, f"seed{seed_idx}")))
            utils.make_dir(run_dir)
            args.run_dir = run_dir
            visualize_env = make_eval_env(args, 4, visualize=True)
            evaluate(args, model, args.final_num_test_seeds, 
                     device, use_render=True, store_traj=True, isFinalEval=True, eval_envs=visualize_env)
            
        break
            
    save_print_result(args, result_list, result_var_list, sparse_result_list, sparse_result_var_list, repeat_num, test_name)

        
def eval_ego_co_player(args, model, co_model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t):
    model_file_names = "model.tar"
    _, co_player_list = get_target_list(overcooked_result_dir, model_file_names)
    
    while True:
        print("\nEgo vs Co_agent")
        print("\n<Ego Agent List>")
        for idx, checkpoint in enumerate(checkpoint_list):
            print(f"[{idx+1}]: {checkpoint[0]}, {checkpoint[1]} seeds")
        
        x = int(input("select num for evaluate. select 0 to close: "))
        if x < 1 or x > len(checkpoint_list):
            print("exit")
            break
          
        print("\n<Co Agent method List>")
        for co_idx, co_checkpoint in enumerate(co_player_list):
            print(f"[{co_idx+1}]: {co_checkpoint[0]}, {co_checkpoint[1]} seeds")
        
        co_x = int(input("select num for evaluate. select 0 to close: "))
        if co_x < 1 or co_x > len(co_player_list):
            print("exit")
            break

        result_list = []
        result_var_list = []
        sparse_result_list = []
        sparse_result_var_list = []
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        
        test_name = f"{checkpoint_list[x-1][0]} vs {co_player_list[co_x-1][0]}"
        print(f"\n{test_name}\n")
        for seed_idx in range(checkpoint_list[x-1][1]):
            model_checkpoint = torch.load(checkpoint_list[x-1][2][seed_idx], map_location=map_location)
            model.load_state_dict(model_checkpoint["model_state_dict"])
            model.to(device)
            seed_result_list = []
            sparse_seed_result_list = []
            
            for co_seed_idx in range(co_player_list[co_x-1][1]):
                print(f"\nego_seed{seed_idx} vs co_seed{co_seed_idx}\n")
                co_agent_model_checkpoint = torch.load(co_player_list[co_x-1][2][co_seed_idx], map_location=map_location)
                co_model.load_state_dict(co_agent_model_checkpoint["model_state_dict"])
                co_model.to(device)
                
                for i in tqdm(range(repeat_num)):
                    if is_e3t:
                        eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate_e3t(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs, num_processes=args.num_processes_test, use_render=False, store_traj=False, co_player=co_model)
                    else:
                        eval_episode_rewards, _, _, eval_episode_sparse_reward = evaluate(args, model, args.final_num_test_seeds, device, eval_envs=eval_envs, co_player=co_model)
                    seed_result_list.append(eval_episode_rewards)
                    sparse_seed_result_list.append(eval_episode_sparse_reward)
            seed_result_list = np.array(seed_result_list)
            sparse_seed_result_list = np.array(sparse_seed_result_list) 
            result_list.append(seed_result_list.mean(axis=0))
            result_var_list.append(seed_result_list.var(axis=0))
            sparse_result_list.append(sparse_seed_result_list.mean(axis=0))
            sparse_result_var_list.append(sparse_seed_result_list.var(axis=0))
            if visualize:
                run_dir = os.path.expandvars(os.path.expanduser("%s/%s/%s/%s" % (args.result_log_dir, args.xpid, test_name, f"seed{seed_idx}")))
                utils.make_dir(run_dir)
                args.run_dir = run_dir
                visualize_env = make_eval_env(args, 2, visualize=True)
                evaluate(args, model, args.final_num_test_seeds, 
                        device, use_render=True, store_traj=True, isFinalEval=True, eval_envs=visualize_env)

        save_print_result(args, result_list, result_var_list, sparse_result_list, sparse_result_var_list, repeat_num, test_name)


def eval_saved_model(args, checkpoint_list, device, test_method, repeat_num=50, visualize=False, is_e3t=True):   

    eval_envs = make_eval_env(args=args, test_method=test_method)
    if is_e3t:
        model = OvercookedPolicy_E3T(eval_envs.observation_space.shape, eval_envs.action_space.n, args).to(device)
        co_model = OvercookedPolicy_E3T(eval_envs.observation_space.shape, eval_envs.action_space.n, args).to(device)
    else:
        model = OvercookedPolicy(eval_envs.observation_space.shape, eval_envs.action_space.n, args)
        co_model = OvercookedPolicy(eval_envs.observation_space.shape, eval_envs.action_space.n, args)
    
    if test_method == 1:
        eval_ego_proxy(args, model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t)
    elif test_method == 2:
        eval_ego_ego(args, model, co_model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t)
    elif test_method == 3:
        eval_ego_co_player(args, model, co_model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t)
    else:
        eval_proxy_proxy(args, model, checkpoint_list, device, repeat_num, visualize, eval_envs, is_e3t)
        

if __name__ == "__main__":
    args = parser.parse_args()
    args.obp_eval_map = True
    args.is_save_model_test = True
    layout_path = os.path.join(args.layouts_dir, args.layouts_type)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')
    
    test_method, test_name = select_test_method()
    
    overcooked_result_dir = "/data/overcooked_plr/overcooked_result/scientific_reports/"
    args.xpid = "lr-%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), test_name)

    _, checkpoint_list = get_target_list(overcooked_result_dir, "model.tar")

    eval_saved_model(args, checkpoint_list, device, test_method, repeat_num=1, visualize=False, is_e3t=False)
