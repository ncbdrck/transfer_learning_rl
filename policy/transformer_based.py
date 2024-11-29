import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Compute the positional encodings once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Inputs:
            x: Tensor of shape (seq_len, batch_size, embedding_dim)
        Outputs:
            x: Tensor with positional encodings added
        """
        x = x + self.pe[:x.size(0)]
        return x

# Transformer-based Actor Network
class ActorTransformer(nn.Module):
    def __init__(self, env_params, nhead=8, num_layers=2, dropout=0.1):
        super(ActorTransformer, self).__init__()
        self.max_action = env_params['action_max']
        input_dim = env_params['obs'] + env_params['goal']
        self.embedding_dim = 256  # Embedding dimension

        # Embedding layer
        self.embedding = nn.Linear(input_dim, self.embedding_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=self.embedding_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.action_out = nn.Linear(self.embedding_dim, env_params['action'])

    def forward(self, x):
        """
        Inputs:
            x: Tensor of shape (batch_size, input_dim)
        Outputs:
            actions: Tensor of shape (batch_size, action_dim)
        """
        # Add a sequence dimension (seq_len=1)
        x = x.unsqueeze(0)  # Shape: (1, batch_size, input_dim)
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        # Transformer expects input of shape (seq_len, batch_size, embedding_dim)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Remove sequence dimension: Shape: (batch_size, embedding_dim)
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

# Transformer-based Critic Network
class CriticTransformer(nn.Module):
    def __init__(self, env_params, nhead=8, num_layers=2, dropout=0.1):
        super(CriticTransformer, self).__init__()
        self.max_action = env_params['action_max']
        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.embedding_dim = 256  # Embedding dimension

        # Embedding layer
        self.embedding = nn.Linear(input_dim, self.embedding_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=self.embedding_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.q_out = nn.Linear(self.embedding_dim, 1)

    def forward(self, x, actions):
        """
        Inputs:
            x: Tensor of shape (batch_size, input_dim_without_action)
            actions: Tensor of shape (batch_size, action_dim)
        Outputs:
            q_value: Tensor of shape (batch_size, 1)
        """
        # Concatenate inputs and normalize actions
        x = torch.cat([x, actions / self.max_action], dim=1)  # Shape: (batch_size, input_dim)
        # Add a sequence dimension
        x = x.unsqueeze(0)  # Shape: (1, batch_size, input_dim)
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        # Transformer processing
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Shape: (batch_size, embedding_dim)
        q_value = self.q_out(x)
        return q_value
