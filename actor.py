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
        self.expl_type = info['expl_type']
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
    
    def get_dist(self, state, std=1.0, min_std=0.1, std_bias=1e-3):
        logits = self.model(state)
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        
        if self.type_dist == 'one_hot':
            return OneHotCategorical(logits=logits)
        elif self.type_dist == 'normal':
            
            mean = torch.tanh(self.mean_head(logits))
            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            
            logstd = self.logstd_head(logits)
            logstd = torch.nan_to_num(logstd, nan=0.0, posinf=2.0, neginf=-5.0)
            
            std = nn.functional.softplus(logstd) + min_std
     
            dist = Normal(mean, std)
            dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
            return SampleDist(Independent(dist, 1))

    
    def forward(self, state):
        action_dist = self.get_dist(state)
        action = action_dist.rsample()
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        return action, action_dist
    
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.SiLU(), 
            nn.RMSNorm(hidden),
            
            nn.Linear(hidden, hidden), 
            nn.SiLU(), 
            nn.RMSNorm(hidden),
            
        )
        self.head = nn.Linear(hidden, 2 * self.out_dim)
    def forward(self, x): 
        h = self.net(x)
        params = self.head(h)                           # [N, 2*out_dim]
        mean, log_std = params.split(self.out_dim, dim=-1)
        # clamp for stability; exp is simple and works well here
        log_std = log_std.clamp(-5.0, 2.0)
        std = log_std.exp() + 1e-6
        
        return Independent(Normal(mean, std), 1)
    
class Plan2Explore(nn.Module):
    def __init__(self, state_size, k=5, hidden=512, fixed_std=1.0, device="mps"):
        super().__init__()
        self.state_size = state_size
        self.heads = nn.ModuleList([MLP(self.state_size, self.state_size, hidden) for _ in range(k)])

        self.to(device)     
    def _intrinsic_reward(self, state):
        
        # [N, F+A]
        preds = torch.cat([head(state).mean[None] for head in self.heads], 0)  # [K, N, F]
        
        disag = torch.log(torch.mean(torch.std(preds, 0), -1)[..., None])

        return disag
    
    def loss(self, state):
        self.to(state.device)  
        target = state[:-1]
        next_state = state[1:].detach()
        preds = [head(target) for head in self.heads]
        
        likes = torch.cat([torch.mean(pred.log_prob(next_state))[None] for pred in preds], 0 )
        
        loss = -torch.mean(likes)
        
        return loss

