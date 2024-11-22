import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            # we are not applying HER to all the transitions, but only to a subset of them.
            # Calculate the probability that a sampled transition will be relabeled with HER.
            # For example, if replay_k is 4, future_p will be 0.8, meaning 80% of the time the sampled transition will be relabeled with HER.
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            # if the replay strategy is not future, then we don't apply HER at all
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]  #  we need to know how may timesteps we have in an episode (fixed time steps)
        rollout_batch_size = episode_batch['actions'].shape[0]  # the number of rollouts in the current episode
        batch_size = batch_size_in_transitions  # the number of transitions we want to return

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)  # select a random batch of rollouts of shape (batch_size,)
        t_samples = np.random.randint(T, size=batch_size)  # select a random batch of timesteps of shape (batch_size,)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # so from each episode (rollout) we are selecting a sample at the time step t_samples (one sample per episode)
        # select the states, actions, next states, rewards, etc. of the selected rollouts and timesteps

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        # select the indexes of the transitions that will be relabeled with HER - only returns locations of True values in the array.
        # eg. [1, 3, 5] we only select the 1st, 3rd, and 5th transitions to be relabeled with HER

        # Compute future offsets for HER relabeling.
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        # (T - t_samples) is the number of timesteps left in the episode after the current timestep
        # So we are selecting a random future timestep to be used to replace the goal ( we can use achieved goal of that step as goal)
        # We multiply by a random number between 0 and 1 to select a random future timestep within the remaining timesteps.
        future_offset = future_offset.astype(int)  # we need it to be an integer

        future_t = (t_samples + 1 + future_offset)[her_indexes]  # the timesteps that will be used to replace the goal
        # Here we add 1 because the offset can be zero, so we need to make sure that the future timestep is at least one step ahead
        # so out of all the t_samples, we only care about the ones that are selected to be relabeled with HER
        # we use the her_indexes to select the future_t for those transitions
        # future means we assume the goal is achieved at some future time step

        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]  # we get the achieved goal at the future timestep
        transitions['g'][her_indexes] = future_ag  # we replace the original goals with the achieved future goals

        # to get the params to re-compute reward
        info = {"is_her": True}  # Add this line to create the info dictionary with the is_her flag (only for multiros envs)
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], info), 1)  # recompute the reward
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}  # Reshape transitions to match batch size

        return transitions
