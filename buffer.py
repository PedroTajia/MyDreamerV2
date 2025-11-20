import torch
import numpy as np

import torch 
from tensordict import TensorDict
# from torchrl.collectors.utils import split_trajectories
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SliceSampler



class EpisodicBuffer():
    def __init__(self, capacity, obs_shape, action_shape, seq_len, batch_size, device='cpu', dtype=torch.float32):
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
                       end_key=("done"),
                       strict_length=True))
        
    @torch.no_grad()
    def add(self, obs, action, reward, done):
        obs_t = obs.detach().to(self.dtype).reshape(*self.obs_shape).contiguous().clone()
        # next_obs_t = next_obs.detach().to(self.dtype).reshape(*self.obs_shape).contiguous().clone()

        act_t = action.detach().to(self.dtype).reshape(self.action_shape).contiguous().clone()
        
        rew_t = torch.tensor([float(reward)], dtype=self.dtype)
        done_t = torch.tensor([done], dtype=torch.bool)
      
        td = TensorDict(
            {
                "obs":obs_t.unsqueeze(0),
                # ("next", "obs"): next_obs_t.unsqueeze(0),
                "action":act_t.unsqueeze(0),
                "reward":rew_t,
                "done":done_t,
                # ("next", "done"): done_t.clone()
                
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
        # next_obs = td.get(("next", "obs")) 
        actions = td.get("action")
        rewards = td.get("reward")
        # done = td.get(("next", "done"))
        done = td.get("done")

        if done.ndim > 2:
        # If last dim is 1, squeeze it; if it's B*L (buggy), take the first channel
            done = done.squeeze(-1) if done.shape[-1] == 1 else done[..., 0]
        
        
        obs = obs[:, 1:].to(self.device).permute(1, 0, *range(2, obs.ndim))
        # past_obs = obs[:, :-1].to(self.device).permute(1, 0, *range(2, obs.ndim))
        actions = actions[:, :self.seq_len].to(self.device).permute(1, 0, 2)
        rewards = rewards[:, :self.seq_len].to(self.device).permute(1, 0)
        done = done[:, :self.seq_len].to(self.device).unsqueeze(-1).permute(1, 0, 2)
     

        return obs, actions, rewards, done
        