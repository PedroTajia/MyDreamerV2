from torch import optim
from torch import distributions
from torch import no_grad
from torch import nn
import numpy as np
import imageio
import torch

import importlib, image_codec, actor, buffer, rssm, networks

importlib.reload(image_codec)
from image_codec import Decoder, Encoder

importlib.reload(actor)
from actor import ActorModel

importlib.reload(buffer)
from buffer import Buffer, EpisodicBuffer

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
        


class Train():
    def __init__(self, info, device='mps'):
        super().__init__()
        self.info = info
        self.kl_info = info['kl_info']
        self.loss_info = info['loss_info']
        
        self.device = device
        self._model_init()
        self._optim_initialize()
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
        
        post_state = self.rssm.get_state(posterior)
        
        obs_dist = self.decoder(post_state[:-1])
        reward_dist = self.rewardModel(post_state[:-1])
        cont_dist = self.discountModel(post_state[:-1])
        obs_loss = self.obs_loss(obs_dist, obs[:-1])
        reward_loss = self.reward_loss(reward_dist, reward[1:].unsqueeze(-1))
        cont_loss = self.cont_loss(cont_dist, nonterm[1:])
        
        prior_dist, post_dist, kl_loss = self.kl_loss(prior, posterior)
        total_loss = self.loss_info['kl_scale'] * kl_loss + reward_loss + obs_loss + self.loss_info['discount_scale'] * cont_loss
        
        perpix_mse = (obs[:-1] - obs_dist.sample()).pow(2).mean().item()
        return total_loss, kl_loss, reward_loss, cont_loss, prior_dist, post_dist, posterior, obs_loss, perpix_mse
    
    def train_batch(self):
        obs, actions, rewards, terms = self.buffer.sample()
        nonterms = 1-terms.to(torch.bfloat16)
        
        total_loss, kl_loss, reward_loss, cont_loss, prior_dist, post_dist, posterior, obs_loss, perpix_mse = self.representation_loss(obs, actions, rewards, nonterms)
        self.model_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.get_param(self.world_list), self.info['grad_clip_norm'])
        self.model_optim.step() 
        
        actor_loss, value_loss = self.actorcritic_loss(posterior)
        
        self.actor_optim.zero_grad(set_to_none=True)
        self.value_optim.zero_grad(set_to_none=True)
        
        
        actor_loss.backward()
        value_loss.backward()
        
        nn.utils.clip_grad_norm_(self.get_param([self.actorModel]), self.info['grad_clip_norm'])
        nn.utils.clip_grad_norm_(self.get_param([self.valueModel]), self.info['grad_clip_norm'])
        
        self.actor_optim.step()
        self.value_optim.step()
        return (total_loss, kl_loss, reward_loss, cont_loss, prior_dist, post_dist, obs_loss, perpix_mse, actor_loss, value_loss)
        
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
        # self.encoder = torch.compile(self.encoder)

        
        self.modelstate = self.stoch_size + self.deter_size
        self.decoder = Decoder(obs_shape, self.modelstate, self.info["depth"], self.info["kernel"]).to(self.device)

        self.ret_norm = RetNorm()
        
        
        

    def get_param(self, modules):
        model_params = []
        for module in modules:
            model_params += list(module.parameters())
            
        return model_params
    
    def actor_loss(self, imag_reward, imag_value, discount_, imag_log_prob, policy_entropy):
        lambda_returns = self.get_return(imag_reward[:-1], imag_value[:-1], discount_[:-1], boostrap=imag_value[-1], lambda_= self.info['lambda'])
        # self.ret_norm.update(lambda_returns)
        
        # offset, scale = self.ret_norm.get_stats()
        # returns = ((lambda_returns - offset) / scale).detach()
        # base = (imag_value[:-1] - offset) / scale
        
        # lambda_returns = returns - base
        # reinforce = (((lambda_returns - offset) - imag_value[:-1]).detach() / scale) * imag_log_prob[1:].unsqueeze(-1)
        dynamics = lambda_returns
        
        
        discount = torch.cumprod(torch.cat([torch.ones_like(discount_[:1]), discount_[1:-1]], 0), 0).detach()
        
        
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
    
        actor_loss = -torch.sum(torch.mean(discount * (dynamics + policy_entropy*self.info['policy_entropy_scale']),dim=1))
        
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
            imag_reward = self.rewardModel(imag_modelstate).mean
            imag_value = self.targetValueModel(imag_modelstate).mean
            discount_ = self.info['discount'] * self.discountModel(imag_modelstate).base_dist.probs
        
        actor_loss, discount, lambda_return = self.actor_loss(imag_reward, imag_value, discount_, imag_log_prob, p_entropy)    
        value_loss = self.value_loss(imag_modelstate, discount, lambda_return)
        
        return actor_loss, value_loss
    
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
    
    
    def _optim_initialize(self):
        self.world_list = [self.encoder, self.rssm, self.rewardModel, self.decoder, self.discountModel]
        
        self.actorcritic = [self.actorModel, self.valueModel]
        
        self.model_optim = optim.Adam(self.get_param(self.world_list), self.info['model_lr'])
        self.actor_optim = optim.Adam(self.get_param(self.actorcritic), self.info['actor_lr'])
        self.value_optim = optim.Adam(self.valueModel.parameters(), self.info['value_lr'])
        
        
        
class RetNorm:
    def __init__(self, decay=0.99, device='mps'):
        self.device = device
        self.decay = decay
        self.ema_q = None  # EMA of (P95 - P5)

    @torch.no_grad()
    def update(self, returns):  # returns: [T,B] or [N]
        r = returns.detach().float().reshape(-1)
        
        r_quantile = torch.quantile(input=r, q=torch.tensor([0.05, 0.95], device=self.device))
        
        if self.ema_q is None:
            # initialize EMA with first observation
            self.ema_q = r_quantile.clone()
        else:
            self.ema_q = self.decay * self.ema_q + (1 - self.decay) * r_quantile
    @torch.no_grad()
    def get_stats(self):
        offset = self.ema_q[0]
        scale = torch.clamp(self.ema_q[1] - self.ema_q[0], min=1.0)
        return offset, scale
    