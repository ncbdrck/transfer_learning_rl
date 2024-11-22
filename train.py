import numpy as np
import gymnasium as gym
import os
from maml_td3_arguments import get_args
from mpi4py import MPI
from rl_modules.vanilla.maml_td3_agent import maml_td3_her_agent
import random
import torch

# for environments from panda-gym
import panda_gym

"""
train the agent, the MPI part code is copied from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
- Updated for MAML TD3
- Added support for multiple environments
"""

def get_env_params(env):
    obs, _ = env.reset()
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def launch(args):
    # env_names = ['FetchReach-v2', 'FetchPush-v2', 'FetchPickAndPlace-v2', 'FetchSlide-v2',
    #              'PandaReach-v3', 'PandaPush-v3', 'PandaPickAndPlace-v3', 'PandaSlide-v3']  # all environments
    # env_names = ['FetchPush-v2', 'FetchPickAndPlace-v2', 'FetchSlide-v2']  # fetch environments
    env_names = ['FetchReach-v2']
    # env_names = ['PandaReach-v3', 'PandaPush-v3', 'PandaPickAndPlace-v3', 'PandaSlide-v3']  # panda environments
    # env_names = ['FetchReach-v2', 'FetchPush-v2', 'FetchPickAndPlace-v2', 'FetchSlide-v2',
    #              'PandaReach-v3', 'PandaPush-v3', 'PandaPickAndPlace-v3']

    envs = []
    env_params = []

    # create environments
    for env_name in env_names:
        env = gym.make(env_name)
        envs.append(env)
        env_params.append(get_env_params(env))

    # set random seeds for reproducibility
    seed = args.seed + MPI.COMM_WORLD.Get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        # set the same random seed for all the gpus
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # create the maml_td3_agent to interact with the environments
    maml_td3_trainer = maml_td3_her_agent(args, envs, env_params, env_names, seed)
    maml_td3_trainer.learn()

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    args = get_args()
    launch(args)
