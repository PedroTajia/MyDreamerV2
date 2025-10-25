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


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)    
    
# class DiscDist:
#     def __init__(
#         self,
#         logits,
#         low=-20.0,
#         high=20.0,
#         transfwd=symlog,
#         transbwd=symexp,
#         device="cuda",
#     ):
#         self.logits = logits
#         self.probs = torch.softmax(logits, -1)
#         self.buckets = torch.linspace(low, high, steps=255, device=device)
#         self.width = (self.buckets[-1] - self.buckets[0]) / 255
#         self.transfwd = transfwd
#         self.transbwd = transbwd

#     def mean(self):
#         _mean = self.probs * self.buckets
#         return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

#     def mode(self):
#         _mode = self.probs * self.buckets
#         return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

#     # Inside OneHotCategorical, log_prob is calculated using only max element in targets
#     def log_prob(self, x):
#         x = self.transfwd(x)
#         # x(time, batch, 1)
#         below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
#         above = len(self.buckets) - torch.sum(
#             (self.buckets > x[..., None]).to(torch.int32), dim=-1
#         )
#         # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
#         below = torch.clip(below, 0, len(self.buckets) - 1)
#         above = torch.clip(above, 0, len(self.buckets) - 1)
#         equal = below == above

#         dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
#         dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
#         total = dist_to_below + dist_to_above
#         weight_below = dist_to_above / total
#         weight_above = dist_to_below / total
#         target = (
#             F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
#             + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
#         )
#         log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
#         target = target.squeeze(-2)

#         return (target * log_pred).sum(-1)

#     def log_prob_target(self, target):
#         log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
#         return (target * log_pred).sum(-1)