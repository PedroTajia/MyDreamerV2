import torch
from torch import nn
from torch import distributions
class RSSM(nn.Module):
    def __init__(self, stoch_size, deter_size, actor_size, node_size, embed_size, info, activation=nn.ELU()):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.actor_size = actor_size
        self.activation = activation
        self.node_size = node_size
        self.embed_size = embed_size
        self.category_size = info["category_size"]
        self.class_size = info["class_size"]
        self.mlp_embed_state_action = self.mlp_embed_state_action_()
        self.mlp_prior = self.mlp_prior_()
        self.mlp_post = self.mlp_post_()
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
    def _init_rssm(self, batch_size, device='mps'):
        return (torch.zeros(batch_size, self.deter_size).to(device),
                torch.zeros(batch_size, self.stoch_size).to(device),
                torch.zeros(batch_size, self.stoch_size).to(device))
    def get_state(self, state):
        # deter + stoch
        return torch.cat((state[0], state[1]), dim=-1)
    
    def mlp_embed_state_action_(self):
        return nn.Sequential(nn.Linear(self.stoch_size + self.actor_size, self.deter_size),
                             self.activation)
        
    def mlp_prior_(self):
        return nn.Sequential(nn.Linear(self.deter_size, self.node_size),
                             self.activation,
                             nn.Linear(self.node_size, self.stoch_size))
        
    def mlp_post_(self):
        return nn.Sequential(nn.Linear(self.deter_size + self.embed_size, self.node_size),
                             self.activation,
                             nn.Linear(self.node_size, self.stoch_size))
    
    def rssm_stack_states(self, states, dim=-1):
        return (torch.stack([state[0] for state in states], dim=dim),
                torch.stack([state[1] for state in states], dim=dim),
                torch.stack([state[2] for state in states], dim=dim))
    ##### warning #####
    def stoch_state_(self, x):
        logits = torch.reshape(x, shape=(*x.shape[:-1], self.category_size, self.class_size))
        dist = distributions.OneHotCategorical(logits=logits)
        #Straight-Through
        stoch = dist.sample()
        stoch += dist.probs - dist.probs.detach()
        
        return torch.flatten(stoch, start_dim=-2, end_dim=-1)
    
    def rssm_image(self, prev_action, prev_rssm_state, nonterms=True):
        
        # multiply by nonterm to stop the flow of info when the env is stopped or finish
        # zxa_(t-1) = net[z_(t-1) + a_(t-1)]
        state_action_embed = self.mlp_embed_state_action(torch.cat([prev_rssm_state[1]*nonterms, prev_action], dim=-1))
        
        # h_t = f(z_(t-1), a_(t-1), h_(t-1))
        deter_state = self.rnn(state_action_embed, prev_rssm_state[0] * nonterms)
        
        # get the logits of z_t from deter state (h_t)
        prior_logits = self.mlp_prior(deter_state)
        prior_stoch_state = self.stoch_state_(prior_logits)
        
        return (deter_state, prior_stoch_state, prior_logits)
    
    def rollout_imag(self, horizon, actor, prev_state):
        state = prev_state
        next_state = []
        next_hidden_state = []
        action_entropy = []
        imag_log_probs = []
        
        for _ in range(horizon):
            # from actor([z_t, h_t])
            # stop the flow of gradients when training actor
            action, action_dist = actor(self.get_state(state).detach())
            # get s_t+1 from s_(t)
            state = self.rssm_image(action, state)
            next_state.append(state)
            next_hidden_state.append(state[0])
            action_entropy.append(action_dist.entropy())
            # the probability of that action be in the distibution
            imag_log_probs.append(action_dist.log_prob(action.detach()))
        # get into a sequence of states s_t
        self.next_hidden_state = torch.stack(next_hidden_state, dim=0)
        next_state = self.rssm_stack_states(next_state, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_state, imag_log_probs, action_entropy
    def rssm_obs(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        # only difference is that state z_t is calculated from the obs embedding 
        prior_rssm_state = self.rssm_image(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state[0]
        # [h_t, x_t]
        x = torch.cat([deter_state, obs_embed], dim=-1)
        #f([h_t, x_t])
        posterior_logits = self.mlp_post(x)
        posterior_stoch_state = self.stoch_state_(posterior_logits)
        posterior_rssm_state = (deter_state, posterior_stoch_state, posterior_logits)
        return prior_rssm_state, posterior_rssm_state
    
    def rollout_obs(self, seq_len, obs_embed, action, nonterm, prev_rssm_state):
        prior = []
        posterior = []
        for t in range(seq_len):
            prev_action = action[t]*nonterm[t]
            prior_state, posterior_state = self.rssm_obs(obs_embed[t], prev_action, nonterm[t], prev_rssm_state)
            prior.append(prior_state)
            posterior.append(posterior_state)
            # only use the post z_t to train the models then use the z^_t for imag
            prev_rssm_state = posterior_state
        prior = self.rssm_stack_states(prior, dim=0)
        post = self.rssm_stack_states(posterior, dim=0)
        return prior, post
        