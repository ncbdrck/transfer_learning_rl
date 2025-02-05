import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from utils.mpi_utils.mpi_utils import sync_networks, sync_grads
from utils.mpi_utils.normalizer import normalizer
from utils.replay_buffer import replay_buffer
from utils.her_modules.her import her_sampler
from policy.models import actor, critic
from policy.transformer_based import ActorTransformer, CriticTransformer

# additional imports
from torch.utils.tensorboard import SummaryWriter
import wandb
import time
import pickle
import json
import math

"""
TD3 with HER (MPI-version) for multi-task learning - dynamic gradient updates
- v1
- For envs with different action and observation spaces
- No adaption networks as it was not working as expected (later versions)
- Pick maximum action space and pad the smaller action spaces
- Same for the observation spaces
"""


class TD3_Agent:
    def __init__(self, args, envs, env_params_list, env_names, seed):

        self.log_interval = args.log_interval # for logging the training metrics
        self.learning_starts = args.learning_starts  # learning starts after these many steps
        self.success_rate_calculation_interval = args.success_rate_calculation_interval

        # Individual environment logging as well as for dynamic gradient updates
        self.success_history_env = [[] for _ in range(len(envs))]
        self.ep_len_history_env = [[] for _ in range(len(envs))]
        self.reward_history_env = [[] for _ in range(len(envs))]
        self.global_step_env = [0 for _ in range(len(envs))]
        self.global_step = 0  # for outer loop
        self.update_counter = 0  # for inner loop
        self.task_success_rates = {env_name: [] for env_name in env_names}
        self.easiness_scores = {env_name: [] for env_name in env_names}

        # store done tasks and their task index
        self.done_tasks = []

        # Variables for logging and saving the model
        self.exp_name: str = os.path.basename(__file__)[: -len(".py")]  # Name of the experiment
        self.run_name = f"{args.exp_name}__{self.exp_name}__{args.seed}__{int(time.time())}"
        self.rank = MPI.COMM_WORLD.Get_rank()  # Get the MPI process rank for logging - the main process is rank 0

        # Initialize the Tensorboard Summary Writer and Weights and Biases
        if args.track and self.rank == 0:
            wandb.login()  # for offline mode, remove this line
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )

        # Initialize the Tensorboard Summary Writer
        if self.rank == 0:
            self.writer = SummaryWriter(f"runs/{self.run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

        self.args = args
        self.envs = envs
        self.env_params_list = env_params_list
        self.env_names = env_names
        self._seed = seed

        # find the env with the maximum observation and action space
        self.max_action_dim = max([env_params['action'] for env_params in env_params_list])
        self.max_obs_dim = max([env_params['obs'] for env_params in env_params_list])
        self.env_params = {'obs': self.max_obs_dim,
                           'goal': env_params_list[0]['goal'],
                           'action': self.max_action_dim,
                           'action_max': env_params_list[0]['action_max']
                           }

        # create the network
        if self.args.transformer_agent:
            # create the transformer policy network
            self.actor_network = ActorTransformer(self.env_params)
            self.critic_network1 = CriticTransformer(self.env_params)
            self.critic_network2 = CriticTransformer(self.env_params)
            # build up the target network
            self.actor_target_network = ActorTransformer(self.env_params)
            self.critic_target_network1 = CriticTransformer(self.env_params)
            self.critic_target_network2 = CriticTransformer(self.env_params)
        else:
            # create the policy network
            self.actor_network = actor(self.env_params)
            self.critic_network1 = critic(self.env_params)
            self.critic_network2 = critic(self.env_params)
            # build up the target network
            self.actor_target_network = actor(self.env_params)
            self.critic_target_network1 = critic(self.env_params)
            self.critic_target_network2 = critic(self.env_params)

        # Load the model if specified
        global_step = 0  # initialize the global step to 0
        if self.args.load_model and self.rank == 0:
            # models must be saved in the runs/{run_name} directory
            load_path = f"runs/{args.load_run_name}/{args.load_model_name}"
            # check if the file exists
            if os.path.exists(load_path):
                # log on the screen
                print("\033[92m" + f"Loading model from {load_path}" + "\033[0m")
                # load the model
                checkpoint = torch.load(load_path)
                # load the weights into the network
                self.actor_network.load_state_dict(checkpoint['actor_state_dict'])
                self.critic_network1.load_state_dict(checkpoint['critic_1_state_dict'])
                self.critic_network2.load_state_dict(checkpoint['critic_2_state_dict'])
                self.actor_target_network.load_state_dict(checkpoint['actor_target_state_dict'])
                self.critic_target_network1.load_state_dict(checkpoint['critic_1_target_state_dict'])
                self.critic_target_network2.load_state_dict(checkpoint['critic_2_target_state_dict'])

                # load the global step
                if self.args.continue_training_log:
                    global_step = checkpoint['global_step']

            else:
                raise FileNotFoundError(f"Model not found at {load_path}")

        # load the global step if only we are continuing the training (need to execute this in all the processes)
        if self.args.continue_training_log and self.args.load_model:

            # broadcast the global step to all the cpus
            global_step = MPI.COMM_WORLD.bcast(global_step, root=0)

            # Set the local global step
            if global_step > 0:
                self.global_step = global_step

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network1)
        sync_networks(self.critic_network2)

        # let's print the number of trainable parameters in the model
        if self.rank == 0:
            actor_parm_cnt = self.count_trainable_parameters(self.actor_network)
            critic_1_parm_cnt = self.count_trainable_parameters(self.critic_network1)
            critic_2_parm_cnt = self.count_trainable_parameters(self.critic_network2)
            print("\033[94m" + f"Actor trainable parameters: {actor_parm_cnt:,}" + "\033[0m")
            print("\033[94m" + f"Critic 1 trainable parameters: {critic_1_parm_cnt:,}" + "\033[0m")
            print("\033[94m" + f"Critic 2 trainable parameters: {critic_2_parm_cnt:,}" + "\033[0m")

        # Sync the target networks
        if self.args.load_model:
            # sync the target networks
            sync_networks(self.actor_target_network)
            sync_networks(self.critic_target_network1)
            sync_networks(self.critic_target_network2)
        else:
            # load the weights into the target networks
            self.actor_target_network.load_state_dict(self.actor_network.state_dict())
            self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
            self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())

        # if you use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network1.cuda()
            self.critic_network2.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network1.cuda()
            self.critic_target_network2.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim1 = torch.optim.Adam(self.critic_network1.parameters(), lr=self.args.lr_critic)
        self.critic_optim2 = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)

        # her sampler
        self.her_modules = [her_sampler(self.args.replay_strategy, self.args.replay_k, env.compute_reward) for env in
                            envs]

        # create the replay buffer
        self.buffers = [replay_buffer(env_params, self.args.buffer_size, her_module.sample_her_transitions)
                              for env_params, her_module in zip(env_params_list, self.her_modules)]

        # create the normalizer
        self.o_norms_list = [normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range) for env_params
                             in env_params_list]
        self.g_norms_list = [normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range) for env_params
                             in env_params_list]

        # Create directory to store the model
        if self.rank == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            self.model_path = os.path.join(self.args.save_dir, self.args.exp_name, self.run_name)
            # create the directory if it doesn't exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        self.model_path_multi =  f"runs/{self.run_name}/{self.exp_name}"

        # let's save all the configurations
        if self.rank == 0:
            if not os.path.exists(self.model_path_multi):
                os.makedirs(self.model_path_multi)
            config_filename = os.path.join(self.model_path_multi, 'config.json')
            with open(config_filename, 'w') as f:
                json.dump(vars(args), f, indent=2)

    def learn(self):
        """
        Train the agent. This is the main method that contains the training loop
        """
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                tasks = self.sample_tasks()
                self.outer_loop(tasks)

                # Evaluate the agent after every cycle
                self.evaluation_and_logging_cycle(cycle, epoch)

        # save the main model
        if self.rank == 0 and self.args.save_model:

            # save the model
            model_save_path = os.path.join(self.model_path_multi, 'model.pt')

            print("\033[92m" + f"Saving the model at {model_save_path}" + "\033[0m")
            torch.save({
                'actor_state_dict': self.actor_network.state_dict(),
                'critic_1_state_dict': self.critic_network1.state_dict(),
                'critic_2_state_dict': self.critic_network2.state_dict(),
                'actor_target_state_dict': self.actor_target_network.state_dict(),
                'critic_1_target_state_dict': self.critic_target_network1.state_dict(),
                'critic_2_target_state_dict': self.critic_target_network2.state_dict(),
            }, model_save_path)

    def sample_tasks(self):
        """
        Sample a list of tasks to loop over. This is used for multitask learning
        :return: return a list of tasks
        """
        # sample multiple tasks
        if self.args.multiple_tasks:
            # sample the specific number of tasks if the number of tasks is less than the number of environments
            if self.args.multi_num_tasks < len(self.envs):
                tasks = np.random.randint(0, len(self.envs), size=self.args.multi_num_tasks).tolist()

                # remove the tasks that are already done
                tasks = [task for task in tasks if task not in self.done_tasks]

                # exit the program if all the tasks are done
                if len(tasks) == 0:
                    print("\033[92m" + "All the tasks are mastered. Exiting the program!" + "\033[0m")
                    exit()

                return tasks
            else:
                # if the number of tasks is greater than the number of environments, sample all the environments (default)
                tasks = list(range(len(self.envs)))

                # remove the tasks that are already done
                tasks = [task for task in tasks if task not in self.done_tasks]

                # exit the program if all the tasks are done
                if len(tasks) == 0:
                    print("\033[92m" + "All the tasks are mastered. Exiting the program!" + "\033[0m")
                    exit()

                return tasks
        else:
            # sample a single task
            return [np.random.randint(0, len(self.envs))]

    def outer_loop(self, tasks):
        """
        We loop over the tasks and update the networks
        :param tasks: List of tasks
        """

        # increase the global step
        self.global_step += 1

        for env_idx in tasks:
            # sample the trajectories
            self.inner_loop(env_idx)

        # Recompute weights each iteration or cycle
        if self.args.use_softmax_weights:
            task_weights = self.compute_task_weights_softmax(tasks)
        else:
            task_weights = self.compute_task_weights(tasks)

        # print the task weights
        if self.rank == 0:
            print(f"Task weights: {task_weights}")

            # add them to the tensorboard and wandb
            for env_name, weight in task_weights.items():
                self.writer.add_scalar(f"task_weights/{env_name}", weight, self.global_step)
                if self.args.track:
                    wandb.log({f"task_weights/{env_name}": weight}, step=self.global_step)

        for _ in range(self.args.n_batches):
            total_actor_loss = torch.tensor(0.0, dtype=torch.float32)
            total_critic_loss1 = torch.tensor(0.0, dtype=torch.float32)
            total_critic_loss2 = torch.tensor(0.0, dtype=torch.float32)

            # increase the update counter
            self.update_counter += 1

            if self.args.cuda:
                total_actor_loss = total_actor_loss.cuda()
                total_critic_loss1 = total_critic_loss1.cuda()
                total_critic_loss2 = total_critic_loss2.cuda()

            # calculate the loss for each environment
            for env_idx in tasks:
                actor_loss, critic_loss1, critic_loss2= self.compute_loss(env_idx)
                if self.args.debug:
                    print(f"env:{env_idx} actor_loss: {type(actor_loss)},critic_loss1: {type(critic_loss1)}, "
                          f"critic_loss2: {type(critic_loss2)}")
                    # print actor_loss so we can check if it has gradient tracking
                    print(f"env:{env_idx} actor_loss: {actor_loss.requires_grad}")
                    print(f"env:{env_idx} critic_loss1: {critic_loss1.requires_grad}")
                    print(f"env:{env_idx} critic_loss2: {critic_loss2.requires_grad}")

                env_name = self.env_names[env_idx]
                w = task_weights.get(env_name, 1.0 / len(tasks))  # default weight is 1/N

                # Accumulate losses
                total_actor_loss += w * actor_loss
                total_critic_loss1 += w * critic_loss1
                total_critic_loss2 += w * critic_loss2

            # # Average the losses across tasks
            # total_actor_loss /= len(tasks)
            # total_critic_loss1 /= len(tasks)
            # total_critic_loss2 /= len(tasks)

            if self.args.debug:
                print(f"update counter: {self.update_counter}, total_critic_loss1: {total_critic_loss1}, "
                      f"total_critic_loss2: {total_critic_loss2}, total_actor_loss: {total_actor_loss}")

            # log the losses for each updates
            if self.rank == 0 and self.update_counter % 10 == 0:
                if self.update_counter % self.args.policy_delay == 0:
                    self.writer.add_scalar("rollout/actor_loss", total_actor_loss, self.update_counter)
                self.writer.add_scalar("rollout/critic_loss1", total_critic_loss1, self.update_counter)
                self.writer.add_scalar("rollout/critic_loss2", total_critic_loss2, self.update_counter)
                if self.args.track:
                    if self.update_counter % self.args.policy_delay == 0:
                        wandb.log({"rollout/actor_loss": total_actor_loss}, step=self.update_counter)
                    wandb.log({"rollout/critic_loss1": total_critic_loss1}, step=self.update_counter)
                    wandb.log({"rollout/critic_loss2": total_critic_loss2}, step=self.update_counter)

            if self.update_counter % self.args.policy_delay == 0:
                # update the actor network
                self.actor_optim.zero_grad()
                total_actor_loss.backward()
                sync_grads(self.actor_network)
                self.actor_optim.step()

            # update the critic_network1
            self.critic_optim1.zero_grad()
            total_critic_loss1.backward()
            sync_grads(self.critic_network1)
            self.critic_optim1.step()

            # update the critic_network2
            self.critic_optim2.zero_grad()
            total_critic_loss2.backward()
            sync_grads(self.critic_network2)
            self.critic_optim2.step()

            # soft update the target networks
            if self.update_counter % self.args.policy_delay == 0:
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network1, self.critic_network1)
                self._soft_update_target_network(self.critic_target_network2, self.critic_network2)

        # soft update the target networks
        if self.global_step % self.args.policy_delay == 0:
            self._soft_update_target_network(self.actor_target_network, self.actor_network)
            self._soft_update_target_network(self.critic_target_network1, self.critic_network1)
            self._soft_update_target_network(self.critic_target_network2, self.critic_network2)

        # check if the task is mastered and remove it from the list
        if self.rank == 0:
            self.update_task_selection(tasks)

    def evaluation_and_logging_cycle(self, cycle, epoch):
        """
        Evaluate the agent and log the results

        :param epoch: The current epoch
        :param cycle: The current cycle
        :return:
        """

        success_rate, reward, ep_len, success_rate_per_env, reward_per_env, ep_len_per_env = self.evaluate_agent()
        if self.rank == 0:
            print(f'[{datetime.now()}] Epoch: {epoch+1}, Cycle : {cycle + 1}, eval success rate: {success_rate:.3f}')
            self.writer.add_scalar("Cycle/eval_success_rate", success_rate, self.global_step)
            self.writer.add_scalar("Cycle/eval_reward", reward, self.global_step)
            self.writer.add_scalar("Cycle/eval_ep_len", ep_len, self.global_step)
            if self.args.track:
                wandb.log({"Cycle/eval_success_rate": success_rate}, step=self.global_step)
                wandb.log({"Cycle/eval_reward": reward}, step=self.global_step)
                wandb.log({"Cycle/eval_ep_len": ep_len}, step=self.global_step)

            # print and log for each environment
            for env_name in self.env_names:
                print(f'[{datetime.now()}] Epoch: {epoch+1}, Cycle : {cycle + 1}, eval success rate for {env_name}: {success_rate_per_env[env_name]:.3f}')
                self.writer.add_scalar(f"Cycle/eval_success_rate_{env_name}", success_rate_per_env[env_name], self.global_step)
                self.writer.add_scalar(f"Cycle/eval_reward_{env_name}", reward_per_env[env_name], self.global_step)
                self.writer.add_scalar(f"Cycle/eval_ep_len_{env_name}", ep_len_per_env[env_name], self.global_step)
                if self.args.track:
                    wandb.log({f"Cycle/eval_success_rate_{env_name}": success_rate_per_env[env_name]}, step=self.global_step)
                    wandb.log({f"Cycle/eval_reward_{env_name}": reward_per_env[env_name]}, step=self.global_step)
                    wandb.log({f"Cycle/eval_ep_len_{env_name}": ep_len_per_env[env_name]}, step=self.global_step)

    def _soft_update_target_network(self, target, source):
        """
        Soft update the target network
        :param target: target network
        :param source: source network
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def compute_loss(self, env_idx):
        """
        Compute the actor and critic loss
        :param env_idx: Index of the environment
        :return: actor_loss, critic_loss1, critic_loss2
        """

        # Sample a batch of transitions
        transitions = self.buffers[env_idx].sample(self.args.batch_size)

        # Pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        # Normalize the inputs
        obs_norm = self.o_norms_list[env_idx].normalize(transitions['obs'])
        g_norm = self.g_norms_list[env_idx].normalize(transitions['g'])
        # Pad the observations to have the same dimension as the maximum observation dimension
        if obs_norm.shape[1] < self.max_obs_dim:
            obs_norm = np.pad(obs_norm, ((0, 0), (0, self.max_obs_dim - obs_norm.shape[1])), 'constant')
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        obs_next_norm = self.o_norms_list[env_idx].normalize(transitions['obs_next'])
        # Pad the observations to have the same dimension as the maximum observation dimension
        if obs_next_norm.shape[1] < self.max_obs_dim:
            obs_next_norm = np.pad(obs_next_norm, ((0, 0), (0, self.max_obs_dim - obs_next_norm.shape[1])), 'constant')
        g_next_norm = self.g_norms_list[env_idx].normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        # Pre-process the actions to have the same dimension as the maximum action dimension
        actions = transitions['actions']
        if actions.shape[1] < self.max_action_dim:
            # pad the actions
            actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - actions.shape[1])), 'constant')
            transitions['actions'] = actions

        # Convert to tensors
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # Compute the target Q-values
        with torch.no_grad():
            noise = (torch.randn_like(actions_tensor) * self.args.policy_noise).clamp(-self.args.noise_clip,
                                                                                      self.args.noise_clip)
            actions_next = (self.actor_target_network(inputs_next_norm_tensor) + noise).clamp(
                -self.env_params_list[env_idx]['action_max'], self.env_params_list[env_idx]['action_max'])
            q_target1 = self.critic_target_network1(inputs_next_norm_tensor, actions_next)
            q_target2 = self.critic_target_network2(inputs_next_norm_tensor, actions_next)
            q_target = torch.min(q_target1, q_target2)
            target_q_value = r_tensor + self.args.gamma * q_target
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # Compute current Q-values and the critic loss
        real_q_value1 = self.critic_network1(inputs_norm_tensor, actions_tensor)
        real_q_value2 = self.critic_network2(inputs_norm_tensor, actions_tensor)
        critic_loss1 = (target_q_value - real_q_value1).pow(2).mean()
        critic_loss2 = (target_q_value - real_q_value2).pow(2).mean()

        # Compute critic loss
        # critic_loss = torch.nn.functional.mse_loss(real_q_value, target_q_value)
        # critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # Compute actor loss
        if self.update_counter % self.args.policy_delay == 0:
            actions_real = self.actor_network(inputs_norm_tensor)
            actor_loss = -self.critic_network1(inputs_norm_tensor, actions_real).mean()
            # Add action regularization to keep the actions in check
            actor_loss += self.args.action_l2 * (actions_real / self.env_params_list[env_idx]['action_max']).pow(
            2).mean()
        else:
            actor_loss = torch.tensor(0.0, dtype=torch.float32)
            if self.args.cuda:
                actor_loss = actor_loss.cuda()

        # log the losses for inner loop updates
        if self.rank ==0:
            if self.update_counter % self.args.policy_delay == 0:
                self.writer.add_scalar(f"env_{env_idx}/actor_loss", actor_loss, self.update_counter)
            self.writer.add_scalar(f"env_{env_idx}/critic_loss1", critic_loss1, self.update_counter)
            self.writer.add_scalar(f"env_{env_idx}/critic_loss2", critic_loss2, self.update_counter)
            if self.args.track:
                if self.update_counter % self.args.policy_delay == 0:
                    wandb.log({f"env_{env_idx}/actor_loss": actor_loss}, step=self.update_counter)
                wandb.log({f"env_{env_idx}/critic_loss1": critic_loss1}, step=self.update_counter)
                wandb.log({f"env_{env_idx}/critic_loss2": critic_loss2}, step=self.update_counter)

        return actor_loss, critic_loss1, critic_loss2

    def inner_loop(self, env_idx):
        """
        Sample trajectories and store them in the replay buffer
        :param env_idx: Index of the environment
        """

        # retrieve the task-specific variables
        env = self.envs[env_idx]

        # Sample trajectories
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(self.args.num_rollouts_per_env):
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            observation, _ = env.reset(seed=self._seed)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']
            ep_reward = 0
            ep_done = False

            # loop over the environment
            for t in range(self.env_params_list[env_idx]['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g, env_idx)
                    pi = self.actor_network(input_tensor)
                    action = self._select_actions(pi, env_idx)
                observation_new, r, term, trunc, info = env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']

                # store the episode
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())

                # update the variables
                obs = obs_new
                ag = ag_new

                # for logging
                self.global_step_env[env_idx] += 1
                ep_reward += r

                # check if the episode is done
                if (term or trunc or t + 1 == self.env_params_list[env_idx]['max_timesteps']) and not ep_done:
                    ep_done = True

                    # log the episode
                    if self.rank == 0:
                        self._log_episode(env_idx, t + 1, ep_reward, info.get('is_success', 0))

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            mb_obs.append(ep_obs)
            mb_ag.append(ep_ag)
            mb_g.append(ep_g)
            mb_actions.append(ep_actions)

        # convert them into np arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)

        # store the episodes
        self.buffers[env_idx].store_episode([mb_obs, mb_ag, mb_g, mb_actions])

        # sync the task mastered list
        self.done_tasks = MPI.COMM_WORLD.bcast(self.done_tasks, root=0)

        # check if this task is not inside the done tasks
        # if we don't do this, updating the normalizer will throw an error since it syncs across all the cpus
        if env_idx not in self.done_tasks:
            # update the normalizer
            self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions], env_idx)

    def _log_episode(self, env_idx, episode_length, episode_reward, is_success):
        """
        Log the episode statistics

        :param env_idx: Index of the environment
        :param episode_length: Length of the episode
        :param episode_reward: Reward of the episode
        :param is_success: Whether the episode was successful or not
        :return: None
        """

        # get env name
        env_name = self.env_names[env_idx]

        # append the task success rate
        self.task_success_rates[env_name].append(is_success)

        # append the episode length and calculate the mean
        self.ep_len_history_env[env_idx].append(episode_length)
        if len(self.ep_len_history_env[env_idx]) >= self.log_interval:
            mean_ep_len = np.mean(self.ep_len_history_env[env_idx][-self.log_interval:])
            self.ep_len_history_env[env_idx] = self.ep_len_history_env[env_idx][-self.log_interval:]
        else:
            mean_ep_len = np.mean(self.ep_len_history_env[env_idx])
        # log the mean episode length
        if self.rank == 0:
            self.writer.add_scalar(f"{env_name}/mean_ep_length", mean_ep_len, self.global_step_env[env_idx])
            if self.args.track:
                wandb.log({f"{env_name}/mean_ep_length": mean_ep_len}, step=self.global_step_env[env_idx])

        # append the episode reward and calculate the mean
        self.reward_history_env[env_idx].append(episode_reward)
        if len(self.reward_history_env[env_idx]) >= self.log_interval:
            mean_reward = np.mean(self.reward_history_env[env_idx][-self.log_interval:])
            self.reward_history_env[env_idx] = self.reward_history_env[env_idx][-self.log_interval:]
        else:
            mean_reward = np.mean(self.reward_history_env[env_idx])
        # log the mean episode reward
        if self.rank == 0:
            self.writer.add_scalar(f"{env_name}/mean_reward", mean_reward, self.global_step_env[env_idx])
            if self.args.track:
                wandb.log({f"{env_name}/mean_reward": mean_reward}, step=self.global_step_env[env_idx])

        # append the success rate and calculate the mean
        self.success_history_env[env_idx].append(is_success)
        if len(self.success_history_env[env_idx]) >= self.log_interval:
            mean_success_rate = np.mean(self.success_history_env[env_idx][-self.log_interval:])
            self.success_history_env[env_idx] = self.success_history_env[env_idx][-self.log_interval:]
        else:
            mean_success_rate = np.mean(self.success_history_env[env_idx])
        # log the mean success rate
        if self.rank == 0:
            self.writer.add_scalar(f"{env_name}/mean_success_rate", mean_success_rate,
                                   self.global_step_env[env_idx])
            if self.args.track:
                wandb.log({f"{env_name}/mean_success_rate": mean_success_rate}, step=self.global_step_env[env_idx])


    def _select_actions(self, pi, env_idx):
        """
        Select the actions based on the policy network

        NEW:
            - Slice the action space to match the environment action space

        :param pi:
        :param env_idx:
        :return:
        """
        action = pi.cpu().numpy().squeeze()

        # slice the action space
        action = action[:self.env_params_list[env_idx]['action']]

        # add the gaussian
        action += self.args.noise_eps * self.env_params_list[env_idx]['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params_list[env_idx]['action_max'],
                         self.env_params_list[env_idx]['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params_list[env_idx]['action_max'],
                                           high=self.env_params_list[env_idx]['action_max'],
                                           size=self.env_params_list[env_idx]['action'])
        # choose if you use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        # clip the actions again in case of the random component
        action = np.clip(action, -self.env_params_list[env_idx]['action_max'],
                         self.env_params_list[env_idx]['action_max'])
        return action


    def _preproc_inputs(self, obs, g, env_idx):
        """
        Preprocess the inputs for the networks

        NEW:
            - Pad the observation if the observation space is less than the maximum observation space

        :param obs: observation
        :param g: goal
        :param env_idx: Index of the environment
        :return: inputs_tensor for the networks
        """

        # retrieve the normalizers
        o_norm = self.o_norms_list[env_idx]
        g_norm = self.g_norms_list[env_idx]

        obs_norm = o_norm.normalize(obs)
        g_norm = g_norm.normalize(g)

        # pad the observation if the observation space is less than the maximum observation space
        if obs_norm.shape[0] < self.max_obs_dim:
            obs_norm = np.pad(obs_norm, (0, self.max_obs_dim - obs_norm.shape[0]), 'constant')

        inputs = np.concatenate([obs_norm, g_norm])
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs_tensor = inputs_tensor.cuda()
        return inputs_tensor

    def _update_normalizer(self, episode_batch, env_idx):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        num_transitions = mb_actions.shape[1]
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        her_module = self.her_modules[env_idx]
        transitions = her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre-process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norms_list[env_idx].update(transitions['obs'])
        self.g_norms_list[env_idx].update(transitions['g'])
        # recompute the stats
        self.o_norms_list[env_idx].recompute_stats()
        self.g_norms_list[env_idx].recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    @staticmethod
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_agent(self):
        """
        Evaluate the agent
        :return: mean_success_rate, mean_reward, mean_ep_len, mean_success_rate_per_env, mean_reward_per_env,
        mean_ep_len_per_env

        **** Notice that we are evaluating all the environments even if we are not training on them
        """
        total_success_rate = []
        total_ep_len = []
        total_reward = []

        # for each environment
        total_success_rate_per_env = {env_name: np.float32(0) for env_name in self.env_names}
        total_ep_len_per_env = {env_name: np.float32(0) for env_name in self.env_names}
        total_reward_per_env = {env_name: np.float32(0) for env_name in self.env_names}

        for env_name, env, o_norm, g_norm in zip(self.env_names, self.envs, self.o_norms_list, self.g_norms_list):
            env_success_rate = []
            env_ep_len = []
            env_reward = []

            # get the environment index
            env_idx = self.get_env_idx(env_name)

            for _ in range(self.args.n_test_rollouts):
                observation, _ = env.reset()
                obs = observation['observation']
                g = observation['desired_goal']
                current_ep_len = 0
                current_ep_reward = 0
                done_flag = False

                for t in range(self.env_params_list[env_idx]['max_timesteps']):
                    with torch.no_grad():
                        input_tensor = self._preproc_inputs(obs, g, env_idx)
                        pi = self.actor_network(input_tensor)
                        actions = pi.detach().cpu().numpy().squeeze()
                    observation_new, reward, term, trunc, info = env.step(actions)

                    if not done_flag:
                        current_ep_len += 1
                        current_ep_reward += reward

                        if term or trunc or t + 1 == self.env_params_list[env_idx]['max_timesteps']:
                            done_flag = True
                            env_success_rate.append(info['is_success'])

                    obs = observation_new['observation']
                    g = observation_new['desired_goal']

                env_ep_len.append(current_ep_len)
                env_reward.append(current_ep_reward)

            total_success_rate.append(np.mean(env_success_rate))
            total_ep_len.append(np.mean(env_ep_len))
            total_reward.append(np.mean(env_reward))

            # store the results for each environment
            total_success_rate_per_env[env_name] = np.mean(env_success_rate)
            total_ep_len_per_env[env_name] = np.mean(env_ep_len)
            total_reward_per_env[env_name] = np.mean(env_reward)

        # for all the environments
        local_success_rate = np.mean(total_success_rate)
        if self.rank == 0:
            print(f"local_success_rate: {local_success_rate}")
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        local_ep_len = np.mean(total_ep_len)
        global_ep_len = MPI.COMM_WORLD.allreduce(local_ep_len, op=MPI.SUM)
        local_reward = np.mean(total_reward)
        global_reward = MPI.COMM_WORLD.allreduce(local_reward, op=MPI.SUM)

        mean_success_rate  = global_success_rate / MPI.COMM_WORLD.Get_size()
        mean_ep_len = global_ep_len / MPI.COMM_WORLD.Get_size()
        mean_reward = global_reward / MPI.COMM_WORLD.Get_size()

        # to save each environment's results
        mean_success_rate_per_env = {env_name: np.float32(0) for env_name in self.env_names}
        mean_ep_len_per_env = {env_name: np.float32(0) for env_name in self.env_names}
        mean_reward_per_env = {env_name: np.float32(0) for env_name in self.env_names}
        for env_name in self.env_names:
            local_success_rate_env = total_success_rate_per_env[env_name]
            if self.rank == 0:
                print(f"local_success_rate_env {env_name}: {local_success_rate_env}")
            global_success_rate_env = MPI.COMM_WORLD.allreduce(local_success_rate_env, op=MPI.SUM)
            local_reward_env = total_reward_per_env[env_name]
            global_reward_env = MPI.COMM_WORLD.allreduce(local_reward_env, op=MPI.SUM)
            local_ep_len_env = total_ep_len_per_env[env_name]
            global_ep_len_env = MPI.COMM_WORLD.allreduce(local_ep_len_env, op=MPI.SUM)

            mean_success_rate_per_env[env_name] = global_success_rate_env / MPI.COMM_WORLD.Get_size()
            mean_ep_len_per_env[env_name] = global_ep_len_env / MPI.COMM_WORLD.Get_size()
            mean_reward_per_env[env_name] = global_reward_env / MPI.COMM_WORLD.Get_size()

        return (mean_success_rate, mean_reward, mean_ep_len, mean_success_rate_per_env,
                mean_reward_per_env, mean_ep_len_per_env)

    def save_model(self, env_name: str):
        """
        This lets us save the model for each environment when training multiple environments

        :param env_name:
        :return:
        """
        if self.rank == 0:
            env_idx = self.get_env_idx(env_name)
            torch.save({
                'env_name': env_name,
                'o_norm_mean': self.o_norms_list[env_idx].mean,
                'o_norm_std': self.o_norms_list[env_idx].std,
                'g_norm_mean': self.g_norms_list[env_idx].mean,
                'g_norm_std': self.g_norms_list[env_idx].std,
            }, f"{self.model_path}/norm_{env_name}.pt")

            torch.save({
                'env_name': env_name,
                'actor_state_dict': self.actor_network.state_dict(),
                'critic_1_state_dict': self.critic_network1.state_dict(),
                'critic_2_state_dict': self.critic_network2.state_dict(),
                'actor_target_state_dict': self.actor_target_network.state_dict(),
                'critic_1_target_state_dict': self.critic_target_network1.state_dict(),
                'critic_2_target_state_dict': self.critic_target_network2.state_dict(),
            }, f"{self.model_path}/model_{env_name}.pt")

    def get_env_idx(self, env_name: str):
        """
        get the index of the environment in the list of environments

        :param env_name:
        :return:
        """
        return self.env_names.index(env_name)

    def get_env_name(self, env_idx: int):
        """
        get the name of the environment from the index

        :param env_idx:
        :return:
        """
        return self.env_names[env_idx]

    def compute_task_easiness(self, env_names):
        easiness_scores = {}
        # Consider the average success rate over the last success_rate_calculation_interval episodes

        for env_name in env_names:
            recent_successes = self.task_success_rates[env_name][-self.args.success_rate_calculation_interval:]
            if recent_successes:
                avg_success = sum(recent_successes) / len(recent_successes)
                if self.rank == 0:
                    print(f"env: {env_name}, recent_successes: {recent_successes}, avg_success: {avg_success}")
            else:
                avg_success = 0.0

            if self.rank == 0 and self.args.debug:
                print(f"env: {env_name}, avg_success: {avg_success}")

            # Easiness score = avg_success for now (simple criteria)
            easiness_scores[env_name] = avg_success

            # todo;  store the easiness score - we can use later for task weighting
            self.easiness_scores[env_name].append(avg_success)

            # log the easiness score
            if self.rank == 0:
                self.writer.add_scalar(f"{env_name}/easiness_score", avg_success, self.global_step_env[self.get_env_idx(env_name)])
                if self.args.track:
                    wandb.log({f"{env_name}/easiness_score": avg_success}, step=self.global_step_env[self.get_env_idx(env_name)])
        return easiness_scores

    # Simpler version of the compute_task_weights - Linearly scales the easiness scores to get the weights
    def compute_task_weights(self, tasks):
        env_names = [self.get_env_name(env_idx) for env_idx in tasks]
        easiness = self.compute_task_easiness(env_names)
        total = sum(easiness.values()) + 1e-8  # Avoid division by zero
        weights = {env_name: easiness[env_name] / total for env_name in env_names}

        return weights

    # Compute task weights using softmax - tau_softmax is the temperature parameter
    def compute_task_weights_softmax(self, tasks):
        env_names = [self.get_env_name(env_idx) for env_idx in tasks]
        easiness = self.compute_task_easiness(env_names)
        exp_values = {env_name: math.exp(easiness[env_name] / self.args.tau_softmax) for env_name in env_names}
        total = sum(exp_values.values()) + 1e-8
        weights = {env_name: exp_values[env_name] / total for env_name in env_names}
        return weights

    def is_task_mastered(self, env_name):
        """
        Check if the task is mastered
        """
        recent_successes = self.task_success_rates[env_name][-self.success_rate_calculation_interval * 2:]
        if not recent_successes:
            return False
        avg_success = sum(recent_successes) / len(recent_successes)
        return avg_success > self.args.task_mastered_threshold

    def update_task_selection(self, tasks):
        # get the env names
        env_names = [self.get_env_name(env_idx) for env_idx in tasks]

        # Check mastery criteria
        for env_name in env_names:
            env_idx = self.get_env_idx(env_name)

            # check if the task is already added to the done tasks list
            is_task_done = env_idx in self.done_tasks

            if not is_task_done and self.is_task_mastered(env_name):
                # add the task to the done tasks list
                self.done_tasks.append(env_idx)

                # sync the done tasks list across all the processes
                self.done_tasks = MPI.COMM_WORLD.bcast(self.done_tasks, root=0)

                if self.rank == 0:
                    print(f"Task {env_name} is mastered!")
                self.save_model(env_name)

