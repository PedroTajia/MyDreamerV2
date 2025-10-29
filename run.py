import numpy as np
import imageio
import gymnasium as gym
import time
import torch
import os
from collections import deque
from env import RobotSuiteEnv
from train import Train
from tqdm import tqdm



import wandb

def log_wandb_video_from_hw3(frames_hw3_list, step, key="media/rollout", fps=20):
    """
    frames_hw3_list: list of np.uint8 frames shaped (H, W, 3) in RGB
    Logs directly to W&B without saving any file locally.
    """
    if not frames_hw3_list:
        return
    # Stack -> THWC
    frames_thwc = np.stack(frames_hw3_list, axis=0).astype(np.uint8)           # [T, H, W, 3]
    

#     W&B expects NumPy videos as (T, C, W, H) by default (yes, W then H)
    frames_tchw = np.transpose(frames_thwc, (0, 3, 1, 2))
    # wandb.log({key: wandb.Video(frames_tchw, fps=fps, format="mp4")}, step=step)
    wandb.log(
    {
        "global_step": step,
        key: wandb.Video(frames_tchw, fps=fps, format="mp4"),
    },
    commit=False,  # keep same step with other logs; set True if this is the last log for the step
)

def init_wandb(cfg):
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "mydreamerv2"),
        entity=os.getenv("WANDB_ENTITY", None),
        name=cfg.get("run_name", None),
        config=cfg,                 # your args/hparams dict
        mode=os.getenv("WANDB_MODE", "online"),  # set WANDB_MODE=offline if needed
        id=os.getenv("WANDB_RUN_ID"),            # optional: support resume
        resume=os.getenv("WANDB_RESUME", None),  # e.g., "allow" or "must"
        save_code=True,            # uploads your code for reproducibility
    )
    wandb.run.log_code(".")        # attach current code snapshot
    return run


info = {'deter_size':400,
        'class_size':32, 
        'category_size':32,
        'obs_shape':(3, 64, 64),
        'action_size':7,
        'embedding_size':400,
        'rssm_node_size':600,
        'capacity':int(1e5),
        'seq_len':50,
        'batch_size':32,
        'actor_layer':4,
        'node_size_actor':400,
        'reward_node_size':400,
        'value_node_size':400,
        'discount_node_size':400,
        'depth':32,
        'kernel':4,
        'kl_info':{'alpha':0.8},
        'loss_info':{'discount_scale':1, 'kl_scale':1},
        'model_lr':2e-4,
        'actor_lr':4e-5,
        'value_lr':1e-4,
        'lambda':0.95,
        'policy_entropy_scale':float(3e-3),
        'horizon':15,
        'discount':0.995,
        'grad_clip_norm':100,
        'target_const':1,
        'exp_info':{'train_noise':0.3, 'eval_noise':0.0, 'expl_min':0.1, 'expl_decay':200000.0, 'expl_type':'epsilon_greedy'}}

wandb_run = init_wandb(info)

device = "mps" if torch.backends.mps.is_available( ) else "cpu"
print(f"Device: {device}")



train = Train(info)
env = RobotSuiteEnv(env_name='Lift', robot="Panda", output_obs=info['obs_shape'])
buffer = train.buffer
# diferent part
rssm = train.rssm

prev_state = rssm._init_rssm(1)
prev_action = torch.tensor(env.action_space.sample()).unsqueeze(0).to(device=device)
scores = []
done = False
encoder = train.encoder
decoder = train.decoder
frames = []
frame_rssm = []
actor = train.actorModel

train_ep = 0
obs, score = env.reset(), 0
train_steps = int(5e6)
train_every = 10
seed = 51
target_update = 100


wandb.define_metric("global_step")
wandb.define_metric("*", step_metric="global_step")

for iter in range(1, train_steps):
    train.step = iter
    # training last
    if iter>seed  and iter%train_every == 0:
                total_loss, kl_loss, reward_loss, cont_loss, prior_dist, post_dist, obs_loss, perpix_mse, actor_loss, value_loss = train.train_batch()
    
                wandb.log({
                        "loss/total": float(total_loss.detach()),
                        "loss/reward": float(reward_loss.detach()),
                        "loss/cont": float(cont_loss.detach()),
                        "loss/obs": float(obs_loss.detach()),
                        "loss/actor": float(actor_loss.detach()),
                        "loss/value": float(value_loss.detach()),
                        
                        }, commit=False)
        
    if iter%target_update == 0:
            train.update_target()
        

        
            
    with torch.no_grad():
        obs = obs.to(device, dtype=torch.float32) 
        embed = encoder(obs).unsqueeze(0).to(device)
        
        
        done_t = torch.as_tensor(done, dtype=torch.bool, device=device).view(1, 1)
        cont = (~done_t).to(torch.float32)
        
        
        _, post_state = rssm.rssm_obs(embed, prev_action, cont , prev_state)
        model_state = rssm.get_state(post_state).to(device)
        action, action_dist = actor(model_state)
        action = action.detach()
        scores.append(score)
        frames.append(env.render())
    
    next_obs, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
    
    if done:

        train_ep += 1
        
        buffer.add(next_obs, action.squeeze(0).cpu(), reward, done)
        if len(frames) > 0:
            log_wandb_video_from_hw3(frames, step=iter, key="media/rollout", fps=20)

        frames = []
        
        
        with torch.no_grad():
                rssm.eval(); train.decoder.eval()
                imag_state, imag_log_probs, _= rssm.rollout_imag(info['horizon'] , actor, prev_state)
                
                imag_modelstate = train.rssm.get_state(imag_state)
                obs_dist = train.decoder(imag_modelstate)
                obs_mean = getattr(obs_dist, "mean")  # [T, B, C, H, W]
                obs_mean = obs_mean + 0.5

                for t in range(len(obs_mean)):
                        frame = (obs_mean[t, 0].clamp(0, 1) * 255).to(torch.uint8)
                        frame = frame.permute(1, 2, 0).cpu().numpy() 
                        frame_rssm.append(frame)
                log_wandb_video_from_hw3(frame_rssm, step=iter, key="media/rollout_rssm", fps=20)
        frame_rssm = []
        
        obs, score = env.reset(), 0
        prev_state = rssm._init_rssm(1)
        prev_action = torch.tensor(env.action_space.sample()).unsqueeze(0).to(device=device)
        done = False
        continue
    else:
        buffer.add(obs, action.squeeze(0).cpu(), reward, done)
        obs = next_obs
        prev_state = post_state
        prev_action = action
        score += reward
        
    # wandb.log({
    #     "train/episode_return": score,
        
    #     }, step=iter)
    wandb.log({
    "global_step": iter,
    "train/episode_return": score,
    }, commit=True)
    
print(np.mean(scores))