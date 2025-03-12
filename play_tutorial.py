import argparse
import os
from overcooked_ai_py.env import OverCookedEnv_Play
from overcooked_ai.play_overcooked.userstudy import *
import pygame
from overcooked_ai.play_overcooked.test_policy import OvercookedPolicy
from level_replay.arguments import parser
import torch
from collections import deque

class Workspace(object):
    def __init__(self, args):
        self.args = args
        self.p0_action_buffer = deque()
        self.p1_action_buffer = deque()
    
    def enqueue_huamn_action(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:  # up
                    self.p0_action_buffer.append(0)
                if event.key == pygame.K_s:  # down
                    self.p0_action_buffer.append(1)
                if event.key == pygame.K_a:  # left
                    self.p0_action_buffer.append(3)
                if event.key == pygame.K_d:  # right
                    self.p0_action_buffer.append(2)
                if event.key == pygame.K_LSHIFT:  # pickup
                    self.p0_action_buffer.append(5)
             
                if event.key == pygame.K_UP:  # up
                    self.p1_action_buffer.append(0)
                if event.key == pygame.K_DOWN:  # down
                    self.p1_action_buffer.append(1)
                if event.key == pygame.K_LEFT:  # left
                    self.p1_action_buffer.append(3)
                if event.key == pygame.K_RIGHT:  # right
                    self.p1_action_buffer.append(2)
                if event.key == pygame.K_SPACE:  # pickup
                    self.p1_action_buffer.append(5)
                    
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def run(self):

        device = 'cpu'

        num_processes = 1
        env = OverCookedEnv_Play(scenario=self.args.scenario, episode_length=self.args.epi_length, time_limits=self.args.time_limits, tutorial=True)
        obs = env.reset()
        both_obs, curr_state, other_agent_idx = obs["both_agent_obs"], obs["overcooked_state"], obs["other_agent_env_idx"]
        obs0 = both_obs[:, 0, :, :]
        obs1 = both_obs[:, 1, :, :]

        clock = pygame.time.Clock()
        try:
            # Wait for starting
            image = env.render()
            screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[...,::-1],1))), (0,0))
            pygame.display.flip()

            flag = True
            while flag:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        flag = False

            start_time = time.time()
            while curr_state.timestep < args.epi_length:
                clock.tick(6.67)

                self.enqueue_huamn_action()
                
                if self.p0_action_buffer:
                    p0_action = self.p0_action_buffer.popleft()
                else:
                    p0_action = 4
                    
                if self.p1_action_buffer: 
                    p1_action = self.p1_action_buffer.popleft()
                else:
                    p1_action = 4

                human_action = [p0_action, p1_action]

                obs, reward, done, info = env.step(action=human_action)
                both_obs, curr_state, other_agent_idx = obs["both_agent_obs"], obs["overcooked_state"], obs["other_agent_env_idx"]

                end_time = time.time()
                game_time = end_time - start_time
                image = env.render(game_time)

                screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[...,::-1],1))), (0,0))
                pygame.display.flip()

            print('finish_time : ', game_time)

        finally:
            pygame.quit()



if __name__ == '__main__':
    # And Play!!!
    parser.add_argument('--scenario', type=str, default='simple')
    parser.add_argument('--time_limits', type=int, default=15)
    parser.add_argument('--epi_length', type=int, default=100)
    parser.add_argument('--working_directory', type=str, default='./overcooked_ai/play_overcooked/result')
    parser.add_argument('--method', type=str, default='pbt')
    parser.add_argument('--strategy', type=str, default='return')
    parser.add_argument('--seed_num', type=str, default=1)
    args = parser.parse_args()

    os.makedirs(args.working_directory, exist_ok=True)
    workspace = Workspace(args)
    workspace.run()
