import gym
import tqdm
import numpy as np
import datetime
from overcooked_ai_py.utils import mean_and_std_err
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, BASE_REW_SHAPING_PARAMS
from overcooked_ai_py.mdp.overcooked_trajectory import TIMESTEP_TRAJ_KEYS, EPISODE_TRAJ_KEYS, DEFAULT_TRAJ_KEYS
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import os
import imageio.v3
import pickle

DEFAULT_ENV_PARAMS = {
    "horizon": 400
}

MAX_HORIZON = 1e10

class OvercookedEnv(object):
    """An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, mdp, start_state_fn=None, horizon=MAX_HORIZON, debug=False, seed=None):
        """
        mdp (OvercookedGridworld or function): either an instance of the MDP or a function that returns MDP instances
        start_state_fn (OvercookedState): function that returns start state for the MDP, called at each environment reset
        horizon (float): number of steps before the environment returns done=True
        """
        self.level_seed = seed
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")

        self.horizon = horizon
        self.start_state_fn = start_state_fn
        self.reset()
        self.number = 0
        if self.horizon >= MAX_HORIZON and self.state.order_list is None and debug:
            print("Environment has (near-)infinite horizon and no terminal states")

    def __repr__(self):
        """Standard way to view the state of an environment programatically
        is just to print the Env object"""
        return self.mdp.state_string(self.state)

    def print_state_transition(self, a_t, r_t, info):
        print("Timestep: {}\nJoint action taken: {} \t Reward: {} + shape * {} \n{}\n".format(
            self.t, tuple(Action.ACTION_TO_CHAR[a] for a in a_t), r_t, info["shaped_r"], self)
        )

    @property
    def env_params(self):
        return {
            "start_state_fn": self.start_state_fn,
            "horizon": self.horizon
        }

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    @staticmethod
    def print_state(mdp, s):
        e = OvercookedEnv(mdp, s)
        print(e)

    def copy(self):
        return OvercookedEnv(
            mdp=self.mdp.copy(),
            start_state_fn=self.start_state_fn,
            horizon=self.horizon
        )

    def step(self, joint_action):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        self.t += 1
        next_state, mdp_infos = self.mdp.get_state_transition(self.state, joint_action)

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict([{}, {}], mdp_infos)

        if done: self._add_episode_info(env_info)

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
        return next_state, timestep_sparse_reward, done, env_info

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.mdp = self.mdp_generator_fn()
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_category_rewards_by_agent": np.array([[0, 0, 0]] * self.mdp.num_players)
        }
        self.game_stats = {**rewards_dict}

    def is_done(self):
        """Whether the episode is over."""
        if self.horizon == -1:
            return self.mdp.is_terminal(self.state)
        else:
            return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    def _prepare_info_dict(self, joint_agent_action_info, mdp_infos):
        """
        The normal timestep info dict will contain infos specifc to each agent's action taken,
        and reward shaping information.
        """
        # Get the agent action info, that could contain info about action probs, or other
        # custom user defined information
        env_info = {"agent_infos": [joint_agent_action_info[agent_idx] for agent_idx in range(self.mdp.num_players)]}
        # TODO: This can be further simplified by having all the mdp_infos copied over to the env_infos automatically
        env_info["sparse_r_by_agent"] = mdp_infos["sparse_reward_by_agent"]
        env_info["shaped_r_by_agent"] = mdp_infos["shaped_reward_by_agent"]
        env_info["shaped_info_by_agent"] = mdp_infos["shaped_info_by_agent"]
        env_info["phi_s"] = mdp_infos["phi_s"] if "phi_s" in mdp_infos else None
        env_info["phi_s_prime"] = mdp_infos["phi_s_prime"] if "phi_s_prime" in mdp_infos else None
        env_info["level_seed"] = self.level_seed
        return env_info

    def _add_episode_info(self, env_info):
        env_info["episode"] = {
            "ep_game_stats": self.game_stats,
            "ep_sparse_r": sum(self.game_stats["cumulative_sparse_rewards_by_agent"]),
            "ep_shaped_r": sum(self.game_stats["cumulative_shaped_rewards_by_agent"]),
            "ep_sparse_r_by_agent": self.game_stats["cumulative_sparse_rewards_by_agent"],
            "ep_shaped_r_by_agent": self.game_stats["cumulative_shaped_rewards_by_agent"],
            # "ep_category_r_by_agent": self.game_stats["cumulative_category_rewards_by_agent"],
            "ep_length": self.t
        }
        return env_info

    def _update_game_stats(self, infos):
        """
        Update the game stats dict based on the events of the current step
        NOTE: the timer ticks after events are logged, so there can be events from time 0 to time self.horizon - 1
        """
        self.game_stats["cumulative_sparse_rewards_by_agent"] += np.array(infos["sparse_reward_by_agent"])
        self.game_stats["cumulative_shaped_rewards_by_agent"] += np.array(infos["shaped_reward_by_agent"])
        # self.game_stats["cumulative_category_rewards_by_agent"] += np.array(infos["shaped_info_by_agent"])

        '''for event_type, bool_list_by_agent in infos["event_infos"].items():
            # For each event type, store the timestep if it occurred
            event_occurred_by_idx = [int(x) for x in bool_list_by_agent]
            for idx, event_by_agent in enumerate(event_occurred_by_idx):
                if event_by_agent:
                    self.game_stats[event_type][idx].append(self.state.timestep)'''

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display: print("Starting state\n{}".format(self))
        for joint_action in joint_action_plan:
            ##mlp, bc 할때 필요
            self.step(joint_action)
            done = self.is_done()
            if display: print(self)
            if done: break
        successor_state = self.state
        self.reset()
        return successor_state, done

    def run_agents(self, agent_pair, include_final_state=False, display=False, display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t).
        """
        assert self.cumulative_sparse_rewards == self.cumulative_shaped_rewards == 0, \
            "Did not reset environment before running agents"
        trajectory = []
        done = False

        if display: print(self)
        while not done:
            s_t = self.state
            a_t = agent_pair.joint_action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, info = self.step(a_t)
            trajectory.append((s_t, a_t, r_t, done))

            if display and self.t < display_until:
                self.print_state_transition(a_t, r_t, info)

        assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True))

        return np.array(trajectory), self.t, self.cumulative_sparse_rewards, self.cumulative_shaped_rewards

    def get_rollouts(self, agent_pair, num_games, display=False, final_state=False, agent_idx=0, reward_shaping=0.0,
                     display_until=np.Inf, info=True):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took),
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        NOTE: standard trajectories format used throughout the codebase
        """
        trajectories = {
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [], # Individual dense (= sparse + shaped * rew_shaping) reward values
            "ep_dones": [], # Individual done values

            # With shape (n_episodes, ):
            "ep_returns": [], # Sum of dense and sparse rewards across each episode
            "ep_returns_sparse": [], # Sum of sparse rewards across each episode
            "ep_lengths": [], # Lengths of each episode
            "mdp_params": [], # Custom MDP params to for each episode
            "env_params": [] # Custom Env params for each episode
        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = self.run_agents(agent_pair, display=display,
                                                                                       include_final_state=final_state,
                                                                                       display_until=display_until)
            obs, actions, rews, dones = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3]
            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews_sparse + tot_rews_shaped * reward_shaping)
            trajectories["ep_returns_sparse"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)

            self.reset()
            agent_pair.reset()

        mu, se = mean_and_std_err(trajectories["ep_returns"])
        if info: print("Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
            mu, np.std(trajectories["ep_returns"]), se, num_games, np.mean(trajectories["ep_lengths"]))
        )

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        return trajectories


class Overcooked(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.

    NOTE: Observations returned are in a dictionary format with various information that is
    necessary to be able to handle the multi-agent nature of the environment. There are probably
    better ways to handle this, but we found this to work with minor modifications to OpenAI Baselines.

    NOTE: The index of the main agent in the mdp is randomized at each reset of the environment, and
    is kept track of by the self.agent_idx attribute. This means that it is necessary to pass on this
    information in the output to know for which agent index featurizations should be made for other agents.

    For example, say one is training A0 paired with A1, and A1 takes a custom state featurization.
    Then in the runner.py loop in OpenAI Baselines, we will get the lossless encodings of the state,
    and the true Overcooked state. When we encode the true state to feed to A1, we also need to know
    what agent index it has in the environment (as encodings will be index dependent).
    """
    env_name = "Overcooked-v0"

    def __init__(self, all_args, layout_list, seed, thread_num, baseline=False):

        if baseline:
            # NOTE: To prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not) reach, we set the same seed internally to all
            # environments. The effect is negligible, as all other randomness
            # is controlled by the actual run seeds
            np.random.seed(0)
        self.all_args = all_args
        self.random_index = all_args.random_index
        self.agent_idx = 0
        self.other_agent_idx = 1
        self.layout_list = layout_list
        self.store_traj = getattr(all_args, "store_traj", False)
        self.mdp_params = {'layout_name': self.layout_list[seed], 'start_order_list': None}
        self.mdp_params.update({
            "rew_shaping_params": BASE_REW_SHAPING_PARAMS,
            "layouts_dir": os.path.join(all_args.layouts_dir, all_args.layouts_type)
        })
        self.env_params = {'horizon': all_args.episode_length}

        if all_args.obp_eval_map:
            self.mdp_params["layouts_dir"] = "/app/overcooked_layout/eval"


        self.mdp_fn = lambda: OvercookedGridworld.from_layout_name(**self.mdp_params)
        self.base_mdp = self.mdp_fn()
        self.base_env = OvercookedEnv(self.mdp_fn, start_state_fn=None, seed=seed, **self.env_params)
        self.featurize_fn = lambda state: self.base_mdp.lossless_state_encoding(state)  # Encoding obs for PPO
        self.featurize_fn_bc = lambda state: self.base_mdp.featurize_state(state)  # Encoding obs for BC
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

        self.use_render = all_args.use_render
        self.thread_num = thread_num
        self.traj_num = 0

        self.visualizer = StateVisualizer()
        
        if all_args.activate_planner:
            from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
            from overcooked_ai.overcooked_ai_py.agents.agent import GreedyHumanModel
            self.mlp = MediumLevelPlanner.from_pickle_or_compute(
                mdp=self.base_mdp,
                mlp_params=NO_COUNTERS_PARAMS,
                force_compute=False
            )
            self.greedy_human = GreedyHumanModel(self.mlp, wait_prob=all_args.wait_prob)
            self.greedy_human_co = GreedyHumanModel(self.mlp, wait_prob=all_args.wait_prob)
            self.greedy_human_num = all_args.human_proxy_num 
        else:
            self.mlp = None
            self.greedy_human = None

      
    def custom_init(self, base_env, seeding_num, featurize_fn, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines:
            # NOTE: To prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not) reach, we set the same seed internally to all
            # environments. The effect is negligible, as all other randomness
            # is controlled by the actual run seeds
            np.random.seed(seeding_num)
        self.base_env = base_env
        self.mdp = base_env.mdp
        self.featurize_fn = featurize_fn
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.reset()

    def seed(self, new_seed):
        self.mdp_params['layout_name'] = self.layout_list[new_seed]
        self.mdp_fn = lambda: OvercookedGridworld.from_layout_name(**self.mdp_params)
        self.base_mdp = self.mdp_fn()
        self.base_env = OvercookedEnv(self.mdp_fn, start_state_fn=None, **self.env_params)
        self.base_env.level_seed = new_seed
        if self.all_args.activate_planner:
            from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
            from overcooked_ai.overcooked_ai_py.agents.agent import GreedyHumanModel
            filename = self.base_mdp.layout_name + "_am.pkl"
            self.mlp = MediumLevelPlanner.from_action_manager_file(filename)
            self.greedy_human = GreedyHumanModel(self.mlp, wait_prob=self.all_args.wait_prob)
            self.greedy_human_co = GreedyHumanModel(self.mlp, wait_prob=self.all_args.wait_prob)

    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.zeros(obs_shape)
        # high = np.ones(obs_shape) * max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = [agent_action, other_agent_action]
        else:
            joint_action = [other_agent_action, agent_action]
            
        if self.greedy_human is not None:
            joint_action = [agent_action, other_agent_action]
            self.greedy_human.set_agent_index(self.other_agent_idx)
            joint_action[self.other_agent_idx] = self.greedy_human.action(self.base_env.state)
            
            if self.greedy_human_num == 2:
                self.greedy_human_co.set_agent_index(self.agent_idx)
                joint_action[self.agent_idx] = self.greedy_human_co.action(self.base_env.state)          
            
        joint_action = tuple(joint_action)

        if self.store_traj:
            self.traj_to_store["action"].append(joint_action)

        next_state, reward, done, info = self.base_env.step(joint_action)

        if self.store_traj:
            self.traj_to_store["info"].append(info)
            self.traj_to_store["state"].append(self.base_env.state.to_dict())

        if self.use_render:
            state = self.base_env.state
            self.traj["ep_states"][0].append(state)
            self.traj["ep_actions"][0].append(joint_action)
            self.traj["ep_rewards"][0].append(reward)
            self.traj["ep_dones"][0].append(done)
            self.traj["ep_infos"][0].append(info)
            if done:
                self.traj['ep_returns'].append(info['episode']['ep_sparse_r'])
                self.traj["mdp_params"].append(self.base_mdp.mdp_params)
                self.traj["env_params"].append(self.base_env.env_params)
                self.render()

        if done and self.store_traj:
            self._store_trajectory()

        ob_p0, ob_p1 = self.featurize_fn(next_state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)

        obs = {"both_agent_obs": both_agents_ob,
               "overcooked_state": next_state,
               "other_agent_env_idx": 1 - self.agent_idx}


        return obs, reward, done, info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()

        if self.random_index:
            self.agent_idx = np.random.choice([0, 1])
            self.other_agent_idx = 1 - self.agent_idx


        self.mdp = self.base_env.mdp
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)

        if self.use_render:
            self.init_traj()

        if self.store_traj:
            self.traj_to_store = {k: [] for k in set(["state", "action", "info"])}
            self.traj_to_store["state"].append(self.base_env.state.to_dict())

        return {"both_agent_obs": both_agents_ob,
                "overcooked_state": self.base_env.state,
                "other_agent_env_idx": 1 - self.agent_idx}

    def init_traj(self):
        self.traj = {k: [] for k in DEFAULT_TRAJ_KEYS}
        for key in TIMESTEP_TRAJ_KEYS:
            self.traj[key].append([])


    def render(self):
        save_dir = f'{self.all_args.run_dir}/gifs/thread_{self.thread_num}_agent_{self.agent_idx}'
        save_dir = os.path.expanduser(save_dir)
        StateVisualizer().display_rendered_trajectory(self.traj, img_directory_path=save_dir, ipython_display=False)
        for img_path in os.listdir(save_dir):
            img_path = save_dir + '/' + img_path
        imgs = []
        imgs_dir = os.listdir(save_dir)
        imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split('.')[0]))
        for img_path in imgs_dir:
            img_path = save_dir + '/' + img_path
            imgs.append(imageio.v3.imread(img_path))
        imageio.mimsave(save_dir + f'/reward_{self.traj["ep_returns"][0]}.gif', imgs, duration=0.05)
        imgs_dir = os.listdir(save_dir)

        for img_path in imgs_dir:
            if img_path.endswith(".png"):
                img_path = save_dir + '/' + img_path
                if 'png' in img_path:
                    os.remove(img_path)

    def _store_trajectory(self):
        if not os.path.exists(f'{self.all_args.run_dir}/trajs/thread_{self.thread_num}/'):
            os.makedirs(f'{self.all_args.run_dir}/trajs/thread_{self.thread_num}/')
        save_dir = f'{self.all_args.run_dir}/trajs/thread_{self.thread_num}/traj_seed_{self.all_args.seed}.pkl'
        pickle.dump(self.traj_to_store, open(save_dir, 'wb'))