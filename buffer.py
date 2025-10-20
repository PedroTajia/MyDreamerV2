import torch
import numpy as np

import torch 
from tensordict import TensorDict
# from torchrl.collectors.utils import split_trajectories
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SliceSampler
class Buffer():
    def __init__(self, capacity, obs_shape, action_shape, seq_len, batch_size, device):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.device = device
        self.obs = torch.empty((self.capacity, *obs_shape)).to(self.device)
        self.action = torch.empty((self.capacity, action_shape)).to(self.device)
        self.reward = torch.empty((self.capacity, )).to(self.device)
        self.terminal = torch.empty((self.capacity, ), dtype=torch.bool).to(self.device)
        
    def add(self, obs, action, reward, done):
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        
    # def sample_idx(self, length):
    #     n = self.size()                      # how many valid frames are available
    #     if n < length:
    #         raise ValueError(f"Need at least {length} frames to sample, have {n}.")
    #     high = self.capacity if self.full else (n - length + 1)  # high is exclusive
    #     idx = np.random.randint(0, high)     # 0 <= idx < high
    #     idxs = (np.arange(length, dtype=np.int64) + idx) % self.capacity
    #     return idxs

    def sample_idx(self, length):
        valid = False
        while not valid:
            idx = np.random.randint(0,self.capacity if self.full else self.idx - length)
            idxs = np.arange(idx, idx + length) % self.capacity
            
            valid = not self.idx in idxs[1:]
        return idxs
    def get_batch(self, idxs, batch_size, length):
        idxs = idxs.transpose(0, 1).reshape(-1)
        obs = self.obs[idxs]
        return obs.reshape(length, batch_size, *self.obs_shape), self.action[idxs].reshape(length, batch_size, -1), self.reward[idxs].reshape(length, batch_size), self.terminal[idxs].reshape(length, batch_size)
    
    def shift(self, obs, action, reward, terminal):
        obs = obs[1:]
        action = action[:-1]
        reward = reward[:-1]
        terminal = terminal[:-1]
        
        return obs, action, reward, terminal
    
    def sample(self):

        obs, a, rew, term = self.get_batch(torch.from_numpy(np.stack([self.sample_idx(self.seq_len+1) for _ in range(self.batch_size)], 0)).long(), self.batch_size, (self.seq_len+1))
        obs, a, rew, term = self.shift(obs, a, rew, term)
        
        return obs, a, rew, term.unsqueeze(-1)
    
    


class EpisodicBuffer():
    def __init__(self, capacity, obs_shape, action_shape, seq_len, batch_size, device='cpu', dtype=torch.bfloat16):
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.seq_len = seq_len  
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.slice_len = self.seq_len + 1

        self.rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=self.capacity),sampler=SliceSampler(
                       slice_len=self.slice_len, 
                       end_key=("next", "done"),
                       strict_length=True))
        
    @torch.no_grad()
    def add(self, obs, action, reward, done):
        obs_t = obs.detach().to(self.dtype).reshape(*self.obs_shape)
        
        act_t = action.detach().to(self.dtype).reshape(self.action_shape)
        
        rew_t = torch.tensor([float(reward)], dtype=self.dtype)
        done_t = torch.tensor([done], dtype=torch.bool)
        
        td = TensorDict(
            {
                "obs":obs_t.unsqueeze(0),
                "action":act_t.unsqueeze(0),
                "reward":rew_t,
                "done":done_t,
                ("next", "done"): done_t.clone()
                
            },
            batch_size=[1],
            
        )
        
        self.rb.extend(td)
        
    def __len__(self) -> int:
        return len(self.rb)
    @torch.no_grad()
    def sample(self):
        td = self.rb.sample(self.batch_size * self.slice_len).reshape(self.batch_size, self.slice_len)
        
        obs = td.get("obs")
        actions = td.get("action")
        rewards = td.get("reward")
        done = td.get(("next", "done"))
        if done.ndim > 2:
        # If last dim is 1, squeeze it; if it's B*L (buggy), take the first channel
            done = done.squeeze(-1) if done.shape[-1] == 1 else done[..., 0]
        
        
        obs = obs[:, 1:].to(self.device).permute(1, 0, *range(2, obs.ndim))
        
        actions = actions[:, :self.seq_len].to(self.device).permute(1, 0, 2)
        rewards = rewards[:, :self.seq_len].to(self.device).permute(1, 0)
        done = done[:, :self.seq_len].to(self.device).unsqueeze(-1).permute(1, 0, 2)
     

        return obs, actions, rewards, done
        