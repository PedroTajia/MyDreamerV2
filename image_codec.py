import torch
from torch import nn
import torch.distributions as td
import numpy as np
class Encoder(nn.Module):
    def __init__(self, input_shape, embed_dim, depth=32, kernel=4, activation = nn.ELU()):
        super(Encoder, self).__init__()
  
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, depth, kernel, stride=2, padding=1),
            activation,
            nn.Conv2d(depth, depth*2, kernel, stride=2, padding=1),
            activation,
            nn.Conv2d(depth*2, depth*4, kernel, stride=2, padding=1),
            activation,
            nn.Conv2d(depth*4, depth*8, kernel, stride=2, padding=1),
            activation
        )
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, *input_shape)
            conv_embed_dim = self.encoder(dummy_x).view(1, -1).shape[1]
    
        if embed_dim == conv_embed_dim:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(conv_embed_dim, embed_dim)    
            
    def forward(self, obs):
        # obs:[50, 50, 3, H, W]
        batch_shape = obs.shape[:-3] #[50, 50]
        img_shape = obs.shape[-3:] #[3, H, W]
        obs = obs.reshape(-1, *img_shape).to(torch.float32)
        # [50 X 50, 3, H, W]
        # no temporal correlation
        with torch.autocast(device_type="mps", enabled=False):
            embedding = self.encoder(obs)
            
            # [50, 50, 3*H*W]
        embedding = torch.reshape(embedding, (*batch_shape, -1))
        # [50, 50, embed_shape]

        return self.fc(embedding)
    

class Decoder(nn.Module):
    def __init__(self, input_shape, embed_dim, depth=32, kernel=4, activation = nn.ELU()):
        super(Decoder, self).__init__()
        self.input_shape = input_shape
        C, H, W = self.input_shape

        h0 = H // 16
        w0 = W // 16
        
        self.conv_shape = (8*depth, h0, w0)
        # embed_dim: hidden_dim(h) + state(z)
        if embed_dim == np.prod(self.conv_shape):
            self.fc = nn.Identity()
        else:
            # x -> (4 * depth, h0, w0)
            self.fc = nn.Linear(embed_dim, np.prod(self.conv_shape))  
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(depth*8, depth*4, kernel, stride=2, padding=1),
            activation,
            nn.ConvTranspose2d(depth*4, depth*2, kernel, stride=2, padding=1),
            activation,
            nn.ConvTranspose2d(depth*2, depth, kernel, stride=2, padding=1),
            activation,
            nn.ConvTranspose2d(depth, C, kernel,  stride=2, padding=1),
            activation
        )
    
    def forward(self, x):
        batch_size = x.shape[:2] # [49, 50]
        embed_size = x.shape[-1] # [600]
        prod_batch = np.prod(batch_size) # [2450]
        
        x = x.reshape(prod_batch, embed_size) # [2450, 600]
        x = self.fc(x) # [2450, 64 X 4 X 4]
        
        x = x.reshape(prod_batch, *self.conv_shape) # [2450, 64, 4, 4]
        
        x = self.decoder(x) # [2450, 3, 10, 10]
        
        mean = x.reshape((*batch_size, *self.input_shape)) #[49, 50, 3, 10, 10]]
        
        return td.Independent(td.Normal(mean, 1), len(self.input_shape))