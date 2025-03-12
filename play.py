import argparse
from datetime import datetime
import pickle
import os
import copy
from overcooked_ai_py.env import OverCookedEnv_Play
from overcooked_ai.play_overcooked.userstudy import *
import pygame
from overcooked_ai.play_overcooked.test_policy import OvercookedPolicy
from level_replay.arguments import parser
import torch
from level_replay import utils
import threading
from collections import deque


class Workspace(object):
    def __init__(self, args):
        self.args = args
        self.model_action = None
        self.human_action_buffer = deque()

    def agent_action(self, model, obs0, pre_obs0, action_reward1, rnn_hidden, rnn_cells, eval_masks):

        _, action, _, _, _, = model.act(obs0, rnn_hidden, rnn_cells, eval_masks, deterministic=False)
        self.model_action = action

    def enqueue_human_action(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:  # up
                    self.human_action_buffer.append(0)
                if event.key == pygame.K_DOWN:  # down
                    self.human_action_buffer.append(1)
                if event.key == pygame.K_LEFT:  # left
                    self.human_action_buffer.append(3)
                if event.key == pygame.K_RIGHT:  # right
                    self.human_action_buffer.append(2)
                if event.key == pygame.K_SPACE:  # pickup
                    self.human_action_buffer.append(5)
                    
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def run(self):

        layout_list = sorted(os.listdir(args.play_layout))
        layout_name = layout_list[self.args.layout_num].split('.')[0]

        date = time.strftime("%Y-%m-%d-%H%M%S")
        run_dir = os.path.expandvars(os.path.expanduser(
            "%s/%s/%s/%s/%s/%s" % (self.args.result_path, self.args.p_num, args.method, args.strategy, layout_name, date)))

        utils.make_dir(run_dir)

        device = 'cuda' if torch.cuda.is_available() else "cpu"

        sars_t = {
            'start_time': datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
            'layout_name': layout_name,
            'image': [],
            'state': [], 'state_raw': [],
            'action': [],
            'reward': [],
            'info': [],
            'next_state': [], 'next_state_raw': []
        }

        num_processes = 1
        env = OverCookedEnv_Play(scenario=layout_name, episode_length=self.args.epi_length, time_limits=self.args.time_limits, tutorial=False)
        obs = env.reset()
        both_obs, curr_state, other_agent_idx = obs["both_agent_obs"], obs["overcooked_state"], obs["other_agent_env_idx"]
        obs0 = both_obs[:, 0, :, :].to(device)

        model = OvercookedPolicy(env.observation_space.shape, env.action_space.n, args).to(device)

        saved_model_path = os.path.join("../overcooked-level-replay/overcooked_ai/play_overcooked/models/", args.method, args.strategy, 'seed' + str(args.seed), 'model.tar')
        saved_model_checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(saved_model_checkpoint["model_state_dict"])

        rnn_hidden = torch.zeros(num_processes, model.recurrent_hidden_state_size, device=device)
        rnn_cells = torch.zeros(num_processes, model.recurrent_hidden_state_size, device=device)
        eval_masks = torch.ones(num_processes, 1, device=device)

        clock = pygame.time.Clock()
        try:
            # Wait for starting
            image = env.render()
            screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

            flag = True
            while flag:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        flag = False

            state_for_save = np.array(env.base_env.mdp.terrain_mtx, dtype=object)
            state_for_save_raw = copy.deepcopy(env.base_env.state.to_dict())

            start_time = time.time()
            game_flag = 0
            while curr_state.timestep < args.epi_length:
                clock.tick(6.67)

                if not game_flag:
                    action_reward1 = 4 + torch.zeros((num_processes, args.past_length), dtype=torch.long, device=device)
                    pre_obs0 = torch.unsqueeze(obs0, 1).repeat(1, args.past_length, 1, 1, 1).to(device)
                    game_flag = 1

                agent_thread = threading.Thread(target=self.agent_action,
                                                args=(model, obs0, pre_obs0, action_reward1, rnn_hidden, rnn_cells, eval_masks))
                agent_thread.start()
                
                self.enqueue_human_action()
                if self.human_action_buffer:
                    human_action = self.human_action_buffer.popleft()
                else:
                    human_action = 4

                agent_thread.join()

                model_action = self.model_action

                joint_action = [model_action.item(), human_action]
                joint_action = np.array(joint_action)

                obs, reward, done, info = env.step(action=joint_action)
                both_obs, curr_state, other_agent_idx = obs["both_agent_obs"], obs["overcooked_state"], obs["other_agent_env_idx"]
                obs0 = both_obs[:, 0, :, :].to(device)

                pre_obs0 = torch.cat((pre_obs0[:, 1:, :, :, :], obs0.unsqueeze(1)), dim=1)
                action_reward1 = torch.cat((action_reward1[:, 1:], torch.tensor([human_action], device=device).unsqueeze(1)), dim=1)

                next_state_for_save = np.array(env.base_env.mdp.terrain_mtx, dtype=object)
                next_state_for_save = from_charlist_to_int_nparray(next_state_for_save, curr_state.player_positions)
                next_state_for_save_raw = copy.deepcopy(env.base_env.state.to_dict())

                if not(np.array_equal(joint_action, [4, 4])):
                    sars_t['image'].append(image)
                    sars_t['state'].append(state_for_save)
                    sars_t['state_raw'].append(state_for_save_raw)
                    sars_t['action'].append(joint_action)
                    sars_t['reward'].append((info['sparse_r_by_agent'], info['shaped_r_by_agent']))
                    sars_t['info'].append(info)
                    sars_t['next_state'].append(next_state_for_save)
                    sars_t['next_state_raw'].append(next_state_for_save_raw)

                state_for_save = next_state_for_save
                state_for_save_raw = next_state_for_save_raw
                end_time = time.time()
                game_time = end_time - start_time
                image = env.render(game_time)

                screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
                pygame.display.flip()

            print('finish_time : ', game_time)

        finally:
            pygame.quit()

            f_path = os.path.join(run_dir, f'{layout_name}.pkl')
            with open(f_path, 'wb') as f:
                sars_t_cpy = copy.deepcopy(sars_t)
                del sars_t_cpy['image']
                pickle.dump(sars_t_cpy, f)
                del sars_t_cpy

            # save all of data
            # f_path = os.path.join(run_dir, f'{layout_name}_large.pkl')
            # with open(f_path, 'wb') as f:
            #     pickle.dump(sars_t, f)


if __name__ == '__main__':
    # And Play!!!
    parser.add_argument('--layout_num', '-num', type=int, default='1')
    parser.add_argument('--time_limits', type=int, default=60)
    parser.add_argument('--epi_length', type=int, default=400)
    parser.add_argument('--result_path', type=str, default='./overcooked_ai/play_overcooked/result')
    parser.add_argument('--p_num', type=str, default="p3")
    parser.add_argument('--method', '-m', type=str, default='pbt')
    parser.add_argument('--strategy', '-s', type=str, default='return_use_metric')
    parser.add_argument('--seed_num', type=str, default=1)
    parser.add_argument('--play_layout', type=str, default='./overcooked_layout/eval')
    args = parser.parse_args()

    os.makedirs(args.result_path + '/' + args.p_num, exist_ok=True)
    workspace = Workspace(args)
    workspace.run()
