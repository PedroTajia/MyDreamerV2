import torch
import numpy as np
from torch import nn
from torch.distributions import OneHotCategorical, Normal, TransformedDistribution, TanhTransform, Independent
class SampleDist(torch.distributions.Distribution):
    def __init__(self, dist, samples=5, validate_args=None):
        super().__init__(batch_shape=dist.batch_shape, event_shape=dist.event_shape, validate_args=False)
        self.dist, self.samples = dist, samples
    def rsample(self, sample_shape=torch.Size()):
        return self.dist.rsample(sample_shape)
    
    def log_prob(self, value):
        return self.dist.log_prob(value)
    
    def entropy(self):
        a = self.dist.rsample((self.samples,))
        return -self.dist.log_prob(a).mean(0)
        

class ActorModel(nn.Module):
    def __init__(self, layers, node_size, deter_size, stoch_size, action_dim, device, info, activation=nn.ELU(), type_dist='normal'):
        super().__init__()
        self.layers = layers
        self.node_size = node_size
        self.activation = activation
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.action_dim = action_dim
        self.train_noise = info['train_noise']
        self.eval_noise = info['eval_noise']
        self.expl_min = info['expl_min']
        self.expl_decay = info['expl_decay']
        # self.expl_type = info['expl_type']
        self.type_dist = type_dist
        self.device = device
        self.model = self.model_make().to(device)
        
    def model_make(self):
        model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
        model += [self.activation]
        for _ in range(self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.activation]
        self.mean_head = nn.Linear(self.node_size, self.action_dim)
        self.logstd_head = nn.Linear(self.node_size, self.action_dim)
        return nn.Sequential(*model)
    
    def get_dist(self, state, std=1.0, min_std=0.1):
        logits = self.model(state)
        if self.type_dist == 'one_hot':
            return OneHotCategorical(logits=logits)
        else:
            mean = torch.tanh(self.mean_head(logits))

            std = nn.functional.softplus(self.logstd_head(logits)) + min_std
     
            dist = Normal(mean, std)
            dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
            return SampleDist(Independent(dist, 1))

    
    def forward(self, state):
        action_dist = self.get_dist(state)
        action = action_dist.rsample()
        return action, action_dist