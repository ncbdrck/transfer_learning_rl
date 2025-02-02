import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskAwareAdaptation(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, num_tasks, feature_size=128):
        super().__init__()
        self.task_emb = nn.Embedding(num_tasks, 16)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + 16, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, feature_size)
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs, goal, task_id):
        task_emb = self.task_emb(task_id)
        x = torch.cat([obs, goal, task_emb], dim=1)
        features = self.encoder(x)
        return features, self.action_decoder(features)

class SharedActor(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)  # Latent action space
        )

    def forward(self, features):
        return self.net(features)

class SharedCritic(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(feature_size + 256, 512),  # Features + latent action
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, features, latent_action):
        return self.q_net(torch.cat([features, latent_action], dim=1))