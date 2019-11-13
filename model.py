import torch
from torch import nn
from torch.distributions import Categorical


class Model(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value


class Actor(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super().__init__()

        self.l1 = nn.Linear(num_inputs, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        y = self.l1(x)
        y = nn.ReLU()(y)
        y = self.l2(y)
        y = nn.Softmax(dim=-1)(y)

        dist = Categorical(y)

        return dist


class Critic(nn.Module):

    def __init__(self, num_inputs, hidden_size=256):
        super().__init__()

        self.l1 = nn.Linear(num_inputs, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        y = self.l1(x)
        y = nn.ReLU()(y)
        y = self.l2(y)

        return y