import argparse

"""
Here are the param for the training
- Multi-Task leaning with TD3
- We are learning task dynamically
"""


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--exp-name', type=str, default='vanilla_multi_task', help='the experiment name')
    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=100, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=10,
                        help='the rollouts per mpi')  # we did not use this

    # Additional args for logging and saving the model
    parser.add_argument('--track', type=bool, default=False,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default='transfer_learning_rl',
                        help='the name of the Weights and Biases project')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='the name of the Weights and Biases entity')
    parser.add_argument('--save-model', type=bool, default=True,
                        help='if toggled, this experiment will save the model after training in the runs/{run_name}')
    parser.add_argument('--save-replay-buffer', type=bool, default=True,
                        help='if toggled, this experiment will save the replay buffer after training in the runs/{run_name}')
    parser.add_argument('--learning_starts', type=int, default=1000,
                        help='the number of steps before learning starts')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='if toggled, this experiment will load the model from the runs/{run_name}/{model_name}')
    parser.add_argument('--load-run-name', type=str, default=None,
                        help='the name of the run to load the model from')
    parser.add_argument('--load-model-name', type=str, default=None,
                        help='the name of the model to load')
    parser.add_argument('--continue-training_log', type=bool, default=False,
                        help='if toggled, this experiment will log from the stopped global_step')
    parser.add_argument('--load-replay-buffer', type=bool, default=False,
                        help='if toggled, this experiment will load the replay buffer from the runs/{run_name}/replay_buffer.pkl')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='the number of episodes for logging the training statistics')

    # td3 specific args - Jay
    parser.add_argument('--policy-noise', type=float, default=0.2,
                        help='the noise added to the target policy during critic update')
    parser.add_argument('--noise-clip', type=float, default=0.5,
                        help='the noise clip value added to the target policy during critic update')
    parser.add_argument('--policy-delay', type=int, default=2,
                        help='the delay in updating the policy')

    # Additional args for multi-task learning
    parser.add_argument('--multiple_tasks', type=bool, default=True,
                        help='if toggled, this each MPI process will train on multiple different task')
    parser.add_argument('--multi_num_tasks', type=int, default=4, help='the number of tasks to sample for training')
    parser.add_argument('--num-rollouts-per-env', type=int, default=2,
                        help='the number of episodes to sample for each iteration')
    parser.add_argument('--transformer_agent', type=bool, default=False,
                        help='if toggled, this experiment will use the transformer agent')
    parser.add_argument('--success_rate_calculation_interval', type=int, default=10,
                        help='the interval to calculate the success rate')
    parser.add_argument('--tau_softmax', type=float, default=0.9,
                        help='the threshold for the dynamic weight calculation')
    parser.add_argument('--use_softmax_weights', type=bool, default=True,
                        help='if toggled, this experiment will use the softmax for dynamic weight calculation')
    parser.add_argument('--task_mastered_threshold', type=float, default=0.98,
                        help='the threshold to consider the task is mastered')

    # for adaptation based policy - Envs with different action and observation spaces
    parser.add_argument('--feature_size', type=int, default=128, help='the size of the feature vector for adaptation')
    parser.add_argument('--lr_adaptation', type=float, default=0.001, help='the learning rate of the adaptation model')
    parser.add_argument('--default_clip_range', type=float, default=5.0, help='the default clip range for the adaptation model')

    parser.add_argument('--debug', type=bool, default=False, help='if toggled, this experiment will run in debug mode')

    args = parser.parse_args()

    return args
