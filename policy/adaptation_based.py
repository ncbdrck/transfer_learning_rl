import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Adaptation-based policy
- Each environment has its own adaption network
- actor and critic networks are shared across environments
- "act_dim" is the maximum action space of all environments. 
- So we need to pad the action space of environments with smaller action space when calling the critic network
"""

# Adaptation network for task-specific processing
class AdaptationNetwork(nn.Module):
    def __init__(self, env_params, output_size=128):
        super(AdaptationNetwork, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# define the actor network
class Actor(nn.Module):
    def __init__(self, input_size=128, act_dim=7):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, act_dim)

    def forward(self, x, max_action):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = max_action * torch.tanh(self.action_out(x))

        return actions

class Critic(nn.Module):
    def __init__(self, input_size=128, act_dim=7):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions, max_action):
        x = torch.cat([x, actions / max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
