from baselines.common.vec_env import VecEnvWrapper

class RewardShapingEnv(VecEnvWrapper):
    """
    Wrapper for the Baselines vectorized environment, which
    modifies the reward obtained to be a combination of intrinsic
    (dense, shaped) and extrinsic (sparse, from environment) reward"""

    def __init__(self, env, reward_shaping_factor=0.0):
        super().__init__(env)
        self.reward_shaping_factor = reward_shaping_factor
        self.env_name = "Overcooked-v0"

        ### Set various attributes to false, than will then be overwritten by various methods

        # Whether we want to query the actual action method from the agent class, 
        # or we use direct_action. Might change things if there is post-processing 
        # of actions returned, as in the Human Model
        self.use_action_method = False

        # Fraction of self-play actions/trajectories (depending on value of self.trajectory_sp)
        self.self_play_randomization = 0.0
        
        # Whether SP randomization should be done on a trajectory level
        self.trajectory_sp = False

        # Whether the model is supposed to output the joint action for all agents (centralized policy)
        # Joint action models are currently deprecated.
        self.joint_action_model = False

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rew, done, infos = self.venv.step_wait()
        # replace rew with shaped rew
        for env_num in range(self.num_envs):
            dense_reward = sum(infos[env_num]['shaped_r_by_agent'])
            rew = list(rew)
            shaped_rew = rew[env_num] + float(dense_reward) * self.reward_shaping_factor
            rew[env_num] = shaped_rew

            if done[env_num]:
                # Log both sparse and dense rewards for episode
                sparse_ep_rew = infos[env_num]['episode']['ep_sparse_r']
                dense_ep_rew = infos[env_num]['episode']['ep_shaped_r']
                infos[env_num]['episode']['r'] = sparse_ep_rew + dense_ep_rew * self.reward_shaping_factor

        return obs, rew, done, infos

    def update_reward_shaping_param(self, reward_shaping_factor):
        """Takes in what fraction of the run we are at, and determines the reward shaping coefficient"""
        self.reward_shaping_factor = reward_shaping_factor
