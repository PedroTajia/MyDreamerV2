import torch
import numpy as np
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
    
    


class EpisodeBuffer():
    def __init__(self, capacity, obs_shape, action_shape, seq_len, batch_size, device):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.episodes = []
        
        
    def start_episode(self):
        episode = dict(
            obs=[],
            action=[],
            reward=[],
            discount=[]
        )
        
        self.episodes.append(episode)
        return len(self.episodes) - 1
    
    def add(self, episode_id, obs, action, reward, done):
        self.episodes[episode_id]['obs'].append(obs)
        self.episodes[episode_id]['action'].append(action.detach())
        self.episodes[episode_id]['reward'].append(reward)
        
#     def add(self, obs, action, reward, done):
#         self.obs[self.idx] = obs
#         self.action[self.idx] = action
#         self.reward[self.idx] = reward
#         self.terminal[self.idx] = done
#         self.idx = (self.idx + 1) % self.capacity
#         self.full = self.full or self.idx == 0
        


#     def sample_idx(self, length):
#         valid = False
#         while not valid:
#             idx = np.random.randint(0,self.capacity if self.full else self.idx - length)
#             idxs = np.arange(idx, idx + length) % self.capacity
            
#             valid = not self.idx in idxs[1:]
#         return idxs
#     def get_batch(self, idxs, batch_size, length):
#         idxs = idxs.transpose(0, 1).reshape(-1)
#         obs = self.obs[idxs]
#         return obs.reshape(length, batch_size, *self.obs_shape), self.action[idxs].reshape(length, batch_size, -1), self.reward[idxs].reshape(length, batch_size), self.terminal[idxs].reshape(length, batch_size)
    
#     def shift(self, obs, action, reward, terminal):
#         obs = obs[1:]
#         action = action[:-1]
#         reward = reward[:-1]
#         terminal = terminal[:-1]
        
#         return obs, action, reward, terminal
    
#     def sample(self):

#         obs, a, rew, term = self.get_batch(torch.from_numpy(np.stack([self.sample_idx(self.seq_len+1) for _ in range(self.batch_size)], 0)).long(), self.batch_size, (self.seq_len+1))
#         obs, a, rew, term = self.shift(obs, a, rew, term)
        
#         return obs, a, rew, term.unsqueeze(-1)
    