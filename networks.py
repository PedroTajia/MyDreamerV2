from torch import nn
from torch import distributions
class RewardModel(nn.Module):
    def __init__(self, stoch_size, deter_size, node_size, activation=nn.ELU()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stoch_size + deter_size, node_size),
            activation,
            nn.Linear(node_size, node_size),
            activation,
            nn.Linear(node_size, node_size),
            
            nn.Linear(node_size, 1)
        )
        
    def forward(self, input):
        dist = self.model(input)
        return distributions.Independent(distributions.Normal(dist, 1), 1)
    
class DiscountModel(nn.Module):
    def __init__(self, stoch_size, deter_size, node_size, activation=nn.ELU()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stoch_size + deter_size, node_size),
            activation,
            nn.Linear(node_size, node_size),
            activation,
            nn.Linear(node_size, node_size),
            
            nn.Linear(node_size, 1)
        )
        
    def forward(self, input):
        dist = self.model(input)
        return distributions.Independent(distributions.Bernoulli(logits=dist), 1)
    
class ValueModel(nn.Module):
    def __init__(self, stoch_size, deter_size, node_size, activation=nn.ELU()):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stoch_size + deter_size, node_size),
            activation,
            nn.Linear(node_size, node_size),
            activation,
            nn.Linear(node_size, node_size),
            
            nn.Linear(node_size, 1)
        )
    def forward(self, input):
        dist = self.model(input)
        return distributions.Independent(distributions.Normal(dist, 1), 1)