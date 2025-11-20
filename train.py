from torch import optim
from torch import distributions
from torch import no_grad
from torch import nn
from common import linear_schedule, one_state
import torch.nn.functional as F
import numpy as np
import torch
import wandb

import importlib, image_codec, actor, buffer, rssm, networks

importlib.reload(image_codec)
from image_codec import Decoder, Encoder

importlib.reload(actor)
from actor import ActorModel, Plan2Explore

importlib.reload(buffer)
from buffer import EpisodicBuffer

importlib.reload(rssm)
from rssm import RSSM

importlib.reload(networks)
from networks import RewardModel, DiscountModel, ValueModel

class Freeze:
    def __init__(self, modules):
        self.modules = modules
        
        self.param_states = [p.requires_grad for p in self.get_param(modules)]
    
    def get_param(self, modules):
        model_params = []
        for module in modules:
            model_params += list(module.parameters())
            
        return model_params
    
    def __enter__(self):
        for param in self.get_param(self.modules):
            param.requires_grad = False
            
    def __exit__(self, type_, val_, tb_):
        for i, param in enumerate(self.get_param(self.modules)):
            param.requires_grad = self.param_states[i]
        


class Train(nn.Module):
    def __init__(self, info, device='mps'):
        super().__init__()
        self.info = info
        self.kl_info = info['kl_info']
        self.loss_info = info['loss_info']
        
        self.device = device
        self._model_init()
        self._optim_initialize()
        self.step = 0 
        self.std = linear_schedule(self.info["std_schedule"], 0)
    def obs_loss(self, dist, obs):
        return -torch.mean(dist.log_prob(obs))
    def reward_loss(self, dist, reward):
        return -torch.mean(dist.log_prob(reward))

    def cont_loss(self, dist, nonterms):
        return -torch.mean(dist.log_prob(nonterms.float()))

    def _get_dist(self, state):
        shape = state[2].shape
        logits = torch.reshape(state[2], shape=(*shape[:-1], self.category_size, self.class_size))
        return distributions.Independent(distributions.OneHotCategoricalStraightThrough(logits=logits), 1)
    def _rssm_detach(self,state):
        return (state[0].detach(),
                state[1].detach(),
                state[2].detach())
        
    def kl_loss(self, prior, posterior):
        prior_dist = self._get_dist(prior)
        posterior_dist = self._get_dist(posterior)
        alpha = self.kl_info['alpha']
        kl_ptr = torch.mean(distributions.kl_divergence(self._get_dist(self._rssm_detach(posterior)), prior_dist))
        kl_rtp = torch.mean(distributions.kl_divergence(posterior_dist, self._get_dist(self._rssm_detach(prior))))
        
        
        rep_loss = torch.clip(kl_ptr, min=1)
        dyn_loss = torch.clip(kl_rtp, min=1)
        kl_loss = alpha*rep_loss + (1-alpha)*dyn_loss
        
        
        return prior_dist, posterior_dist, kl_loss
    
    
    def representation_loss(self, obs, action, reward, nonterm):
        embedding = self.encoder(obs)
    
        prev_state = self.rssm._init_rssm(self.info['batch_size'])

        prior, posterior = self.rssm.rollout_obs(self.info['seq_len'], embedding, action, nonterm, prev_state)
        
        
        state = self.rssm.get_state(posterior)
        
        reward_dist = self.rewardModel(state[:-1])
        cont_dist = self.discountModel(state[:-1])
        obs_dist = self.decoder(state[:-1])

        obs_loss = self.obs_loss(obs_dist, obs[:-1])
       
        reward_loss = self.reward_loss(reward_dist, reward[1:].unsqueeze(-1))
        cont_loss = self.cont_loss(cont_dist, nonterm[1:])
        
        _, _, kl_loss = self.kl_loss(prior, posterior)
        explore_loss = self.plan2explore.loss(state[:-1].clone().detach())
        
        
        total_loss = self.loss_info['kl_scale'] * kl_loss + reward_loss + obs_loss + self.loss_info['discount_scale'] * cont_loss
        
        
        return total_loss, kl_loss, reward_loss, cont_loss, posterior, obs_loss, explore_loss
    
    def consistency_loss(self, z, next_z):
        
        return F.mse_loss(z, next_z.detach())
    def train_batch(self):
        obs, actions, rewards, terms = self.buffer.sample()
        nonterms = 1-terms.to(torch.float32)
        
        total_loss, kl_loss, reward_loss, cont_loss, posterior, obs_loss, explore_loss = self.representation_loss(obs, actions, rewards, nonterms)
        if self.info['explore_only']:
            self.plan2explore_optim.zero_grad(set_to_none=True)
            explore_loss.backward()
            nn.utils.clip_grad_norm_(self.plan2explore.parameters(), self.info['grad_clip_norm'])
            self.plan2explore_optim.step()
        
        
        
        self.model_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.get_param(self.world_list), self.info['grad_clip_norm'])
        
        
        self.model_optim.step() 
        
        actor_loss, value_loss, imag_reward = self.actorcritic_loss(posterior)
        
        self.actor_optim.zero_grad(set_to_none=True)
        self.value_optim.zero_grad(set_to_none=True)
        
        
        actor_loss.backward()
        value_loss.backward()
        
        nn.utils.clip_grad_norm_(self.get_param([self.actorModel]), self.info['grad_clip_norm'])
        nn.utils.clip_grad_norm_(self.get_param([self.valueModel]), self.info['grad_clip_norm'])
        
        self.actor_optim.step()
        self.value_optim.step()
        
       
        
        return (total_loss, kl_loss, reward_loss, cont_loss, obs_loss, actor_loss, value_loss, explore_loss)
        
    def value_loss(self, imag_modelstate, discount, lambda_return):
        with torch.no_grad():
            value_modelstate = imag_modelstate[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_return.detach()
            
        value_dist = self.valueModel(value_modelstate)
       
        return -torch.mean(value_discount * value_dist.log_prob(value_target).unsqueeze(-1))
    def _model_init(self):
        self.class_size = self.info['class_size']
        self.category_size = self.info['category_size']
        self.stoch_size = self.class_size * self.category_size
        self.deter_size = self.info['deter_size']
        obs_shape = self.info['obs_shape']
        actor_size = self.info['action_size']
        
        embedding_size = self.info['embedding_size']
        rssm_node_size = self.info['rssm_node_size']
        
        # self.buffer = Buffer(self.info['capacity'], obs_shape, actor_size, self.info['seq_len'], self.info['batch_size'], self.device)
        self.buffer = EpisodicBuffer(self.info['capacity'], obs_shape, actor_size, self.info['seq_len'], self.info['batch_size'], self.device)
        self.rssm = RSSM(self.stoch_size, self.deter_size, actor_size, rssm_node_size, embedding_size, self.info).to(self.device)
        # self.rssm = torch.compile(self.rssm)
        
        self.actorModel = ActorModel(self.info['actor_layer'], self.info['node_size_actor'], self.deter_size, self.stoch_size, actor_size, self.device, self.info['exp_info']).to(self.device)
        # self.actorModel = torch.compile(self.actorModel)
        
        self.rewardModel = RewardModel(self.stoch_size, self.deter_size, self.info['reward_node_size']).to(self.device)
        
        self.valueModel = ValueModel(self.stoch_size, self.deter_size, self.info['value_node_size']).to(self.device)
        # self.valueModel = torch.compile(self.valueModel)
        
        self.targetValueModel = ValueModel(self.stoch_size, self.deter_size, self.info['value_node_size']).to(self.device)
        # self.targetValueModel = torch.compile(self.targetValueModel)
        self.targetValueModel.load_state_dict(self.valueModel.state_dict())
        
        self.discountModel = DiscountModel(self.stoch_size, self.deter_size, self.info['discount_node_size']).to(self.device)
        self.encoder = Encoder(obs_shape, embedding_size, self.info["depth"], self.info["kernel"]).to(self.device)
        self.targetencoder = Encoder(obs_shape, embedding_size, self.info["depth"], self.info["kernel"]).to(self.device)
        self.targetencoder.load_state_dict(self.encoder.state_dict())
        # self.encoder = torch.compile(self.encoder)

        
        self.modelstate = self.stoch_size + self.deter_size
        self.decoder = Decoder(obs_shape, self.modelstate, self.info["depth"], self.info["kernel"]).to(self.device)

        self.plan2explore = Plan2Explore(self.modelstate , device=self.device)

        self.ret_norm = RetNorm(device=self.device)
    
    @torch.no_grad()
    def estimate_value(self, z, actions, horizon, nonterm):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            reward = self.plan2explore._intrinsic_reward(self.rssm.get_state(z).detach()) 
            nonterms_t = torch.ones(actions[t].shape[0], 1, device=self.device)

            z = self.rssm.rssm_image(actions[t], z, nonterms_t)
            G += discount * reward
            discount *= self.info["discount"]
        G += discount * self.targetValueModel(self.rssm.get_state(z).detach()).mean()
        return G
    
    def plan(self, embed, action, nonterm, eval_mode, step, t0, prev_state):		
        if step < self.info["seed_steps"] and not eval_mode:
            return (torch.empty(self.info["action_size"], dtype=torch.float32, device=self.device).uniform_(-1, 1).unsqueeze(0))   # <- make it [1, A])
        
        horizon = self.info["horizon"]
        num_pi_trajs = int(self.info["mixture_coef"] * self.info["num_samples"])
        prior, posterior = self.rssm.rssm_obs(embed, action, nonterm, prev_state)
        z = tuple(s.repeat(num_pi_trajs, 1) for s in posterior)        
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.info["action_size"], device=self.device)
        for t in range(horizon):
                pi_actions[t] = self.actorModel(self.rssm.get_state(z))[0]
                nonterms_t = torch.ones(pi_actions[t].shape[0], 1, device=self.device)
                z = self.rssm.rssm_image(pi_actions[t], z, nonterms_t)
        prior = self.rssm.rssm_obs(embed, action, nonterm, prev_state)[0]
        z = tuple(s.repeat(self.info["num_samples"]+num_pi_trajs, 1) for s in prior)        
        mean = torch.zeros(horizon, self.info["action_size"], device=self.device)
        std = 2*torch.ones(horizon, self.info["action_size"], device=self.device)
        
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]
        
        
        for i in range(self.info["iterations"]):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.info["num_samples"], self.info["action_size"], device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)
                # Compute elite actions
            value = self.estimate_value(z, actions, horizon, nonterm).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.info["num_elites"], dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.info["temperature"]*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.info["momentum"] * mean + (1 - self.info["momentum"]) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.info["action_size"], device=std.device)
        return a.unsqueeze(0) 
    

    def get_param(self, modules):
        model_params = []
        for module in modules:
            model_params += list(module.parameters())
            
        return model_params
    
    def actor_loss(self, imag_reward, imag_value, discount_, imag_log_prob, policy_entropy, gradient="dynamics"):
        lambda_returns = self.get_return(imag_reward[:-1], imag_value[:-1], discount_[:-1], boostrap=imag_value[-1], lambda_= self.info['lambda'])
        self.register_buffer("ema_q", torch.zeros(2, device=self.device))

        offset, scale = self.ret_norm.update(lambda_returns, self.ema_q)
        
        returns = ((lambda_returns - offset) / scale)
        base = (imag_value[:-1] - offset) / scale
        
        lambda_returns = returns - base
        if gradient == "dynamics":
            target = lambda_returns
        elif gradient == "reinforce":
            target = (((lambda_returns - offset) - imag_value[:-1]).detach() / scale) * imag_log_prob[1:].unsqueeze(-1)
       
        
        
        discount = torch.cumprod(torch.cat([torch.ones_like(discount_[:1]), discount_[1:-1]], 0), 0 ).detach()
        
        
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
    
        # actor_loss = -torch.sum(torch.mean(discount * (reinforce + policy_entropy*self.info['policy_entropy_scale']),dim=1))
        actor_loss = -torch.sum(torch.mean(discount * (target + policy_entropy*self.info['policy_entropy_scale']),dim=1))

        # wandb.log({
        #     "actor/entropy": float(policy_entropy.mean().detach()),
        #     "actor/adv_mean": float(lambda_returns.mean().detach()),
        #     "actor/adv_std":  float(lambda_returns.std().detach()),
        # }, step=self.step)
        
        return actor_loss, discount, lambda_returns
    def _rssm_seq_batch(self, state, seq_len):
        return (
            state[0][:seq_len].flatten(0, 1),
            state[1][:seq_len].flatten(0, 1),
            state[2][:seq_len].flatten(0, 1)
        )
        
   
        
    
    def actorcritic_loss(self, post):
        batch_posterior = self._rssm_detach(self._rssm_seq_batch(post, self.info['seq_len']-1))
       
            
        with Freeze(self.world_list):
            imag_state, imag_log_prob, p_entropy = self.rssm.rollout_imag(self.info['horizon'], self.actorModel, batch_posterior)
        imag_modelstate = self.rssm.get_state(imag_state)
        
        with Freeze(self.world_list + [self.valueModel, self.targetValueModel, self.discountModel]):
            imag_reward = self.rewardModel(imag_modelstate).mean()
            imag_value = self.targetValueModel(imag_modelstate).mean()
            discount_ = self.info['discount'] * self.discountModel(imag_modelstate).base_dist.probs
        with torch.no_grad():
            intrinsic = self.plan2explore._intrinsic_reward(imag_modelstate) 
        if self.info["explore_only"] == True :
            
            imag_reward = intrinsic
        actor_loss, discount, lambda_return = self.actor_loss(imag_reward, imag_value, discount_, imag_log_prob, p_entropy)    
        value_loss = self.value_loss(imag_modelstate, discount, lambda_return)
        
        return actor_loss, value_loss, imag_reward
    
    def get_return(self, reward, value, discount, boostrap, lambda_):
        next_val = torch.cat([value[1:], boostrap.unsqueeze(0)])
        target = reward + discount * next_val * (1-lambda_)
        out = []
        acc_reward = boostrap
        for t in range(reward.shape[0]-1, -1, -1):
            input_ = target[t]
            discount_ = discount[t]
            acc_reward = input_ + discount_ * lambda_ * acc_reward
            out.append(acc_reward)
            
        return torch.flip(torch.stack(out), [0])
    
    def update_target(self):
        for param, target_param in zip(self.valueModel.parameters(), self.targetValueModel.parameters()):
            target_param.data.copy_(self.info['target_const'] * param.data + (1 - self.info['target_const'])*target_param.data)
    
    def update_encoder(self, tau=0.01):
        with torch.no_grad():
            for p, p_target in zip(self.encoder.parameters(), self.targetencoder.parameters()):
                p_target.data.lerp_(p.data, tau)
    
    def _optim_initialize(self):
        self.world_list = [self.encoder, self.rssm, self.rewardModel, self.decoder, self.discountModel]
        
        
        self.actorcritic = [self.actorModel, self.valueModel]
        
        self.model_optim = optim.Adam(self.get_param(self.world_list), self.info['model_lr'])

        self.actor_optim = optim.Adam(self.get_param(self.actorcritic), self.info['actor_lr'])
        self.value_optim = optim.Adam(self.valueModel.parameters(), self.info['value_lr'])
        
        self.plan2explore_optim = optim.Adam(self.plan2explore.parameters(), self.info['plan2explore_lr'])

        
	
        
        
class RetNorm:
    def __init__(self, decay=1e-2, device='mps'):
        self.device = device
        self.decay = decay
         # EMA of (P95 - P5)

    @torch.no_grad()
    def update(self, returns, ema_q):  # returns: [T,B] or [N]
        r = returns.detach().float().reshape(-1)
        
        r_quantile = torch.quantile(input=r, q=torch.tensor([0.05, 0.95], device=self.device))
        
        ema_q = self.decay * r_quantile + (1 - self.decay) * ema_q
            
        offset = ema_q[0]
        scale = torch.clamp(ema_q[1] - ema_q[0], min=1.0)
        return offset, scale
    

    