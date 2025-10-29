import torch
from torch import nn
from torch import distributions
class RewardModel(nn.Module):
    def __init__(self, stoch_size, deter_size, node_size, activation=nn.SiLU()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stoch_size + deter_size, node_size),
            activation,
            nn.RMSNorm(node_size),
            
            nn.Linear(node_size, node_size),
            activation,
            nn.RMSNorm(node_size),
            
            nn.Linear(node_size, 1)
        )
        
    def forward(self, input):
        dist = self.model(input)
        return distributions.Independent(distributions.Normal(dist, 1), 1)
    
class DiscountModel(nn.Module):
    def __init__(self, stoch_size, deter_size, node_size, activation=nn.SiLU()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stoch_size + deter_size, node_size),
            activation,
            nn.RMSNorm(node_size),
            
            nn.Linear(node_size, node_size),
            activation,
            nn.RMSNorm(node_size),
            
            nn.Linear(node_size, 1)
        )
        
    def forward(self, input):
        dist = self.model(input)
        return distributions.Independent(distributions.Bernoulli(logits=dist), 1)
    
class ValueModel(nn.Module):
    def __init__(self, stoch_size, deter_size, node_size, activation=nn.SiLU(), type_dist="normal"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stoch_size + deter_size, node_size),
            activation,
            nn.RMSNorm(node_size),
            
            nn.Linear(node_size, node_size),
            activation,            
            nn.RMSNorm(node_size),
            
            nn.Linear(node_size, 1)
        )
        self.type_dist = type_dist
    def forward(self, input):
        dist = self.model(input)
        return self.get_dist(dist)

    def get_dist(self, dist, std=1.0, min_std=0.1):
        
        if self.type_dist == 'normal':
            return distributions.Independent(distributions.Normal(dist, 1), 1)
