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
        'horizon':14,
        'discount':0.995,
        'grad_clip_norm':100,
        'target_const':1,
        'exp_info':{'train_noise':0.3, 'eval_noise':0.0, 'expl_min':0.1, 'expl_decay':200000.0, 'expl_type':'epsilon_greedy'}}



device = "mps" if torch.backends.mps.is_available( ) else "cpu"
print(f"Device: {device}")

###
start = time.time()
def rmean(buf): 
    return float(np.mean(buf))

kl_hist, rew_hist, cont_hist, obs_hist, mse_hist, act_hist, val_hist = [deque(maxlen=100) for _ in range(7)]
ret_hist = deque(maxlen=100)
###

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
obsList = []
actor = train.actorModel

train_ep = 0
obs, score = env.reset(), 0
train_steps = int(5e6)
train_every = 14
train_seed = 500
target_update = 100
bar = tqdm(total=train_steps-1, dynamic_ncols=True)

it = 0
while True:
        path = f'test_{it}'
        if not os.path.exists(path):
                break
        it += 1
os.mkdir(path)
print(f"Folder {path} created successfully")

for iter in range(1, train_steps):
    
    # training last
    if iter>train_seed  and iter%train_every == 0:
                kl_loss, reward_loss, cont_loss, prior_dist, post_dist, obs_loss, perpix_mse, actor_loss, value_loss = train.train_batch()
        #     print("-"*32+"\n")
        #     print(f"kl_loss:{kl_loss}, reward_loss:{reward_loss}, cont_loss:{cont_loss}, obs_loss:{obs_loss}, perpix_mse:{perpix_mse}")
                kl_hist.append(float(kl_loss.detach().item()))
                rew_hist.append(float(reward_loss.detach().item()))
                cont_hist.append(float(cont_loss.detach().item()))
                obs_hist.append(float(obs_loss.detach().item()))
                mse_hist.append(float(perpix_mse))
                act_hist.append(float(actor_loss.detach().item()))
                val_hist.append(float(value_loss.detach().item()))
                bar.set_description_str(
                f"it {iter}/{train_steps}  ep {train_ep}"
                )
                bar.set_postfix({
                "kl":   f"{rmean(kl_hist):.4f}",
                "rew":  f"{rmean(rew_hist):.4f}",
                "cont": f"{rmean(cont_hist):.4f}",
                "obs":  f"{rmean(obs_hist):.1f}",
                "mse":  f"{rmean(mse_hist):.4f}",
                "actor": f"{rmean(act_hist):.4f}",
                "value": f"{rmean(act_hist):.4f}",
                "R̄100": f"{rmean(ret_hist):.2f}",
        })
    if iter%target_update == 0:
            train.update_target()
        
#     if iter%train_every == 0:
#             print(f'{iter}: train ')
#             kl_loss, reward_loss, cont_loss, prior_dist, post_dist, obs_loss, perpix_mse = train.train_batch()
#             print("-"*32+"\n")
#             print(f"kl_loss:{kl_loss}, reward_loss:{reward_loss}, cont_loss:{cont_loss}, obs_loss:{obs_loss}, perpix_mse:{perpix_mse}")
#     if iter%target_update == 0:
#             train.update_target()
        
            
    with torch.no_grad():
        obs = obs.to(device, dtype=torch.float32) / 255.0
        embed = encoder(obs).unsqueeze(0).to(device)
        
        
        done_t = torch.as_tensor(done, dtype=torch.bool, device=device).view(1, 1)
        cont = (~done_t).to(torch.bfloat16)
        
        
        _, post_state = rssm.rssm_obs(embed, prev_action, cont , prev_state)
        model_state = rssm.get_state(post_state).to(device)
        action, action_dist = actor(model_state)
        # action = actor.add_expl(action, iter).detach()
        action = action.detach()
        scores.append(score)
        obsList.append(env.render())
    
    next_obs, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
    if done:
        train_ep += 1
        # print(f"{train_ep}: Done")
        ret_hist.append(float(score))
        bar.write(f"episode {train_ep}: return {score:.2f} (mean100 {rmean(ret_hist):.2f})")

        buffer.add(obs, action.squeeze(0).cpu(), reward, not done)
        obs, score = env.reset(), 0
        done = False
        prev_state = rssm._init_rssm(1)

        # print('NEW VIDEO')
        # print('-'*32)
        # imageio.mimwrite(f"trainer_{iter}.mp4", obsList, fps=16)
        if len(obsList) > 0:
            try:
                imageio.mimwrite(f"{path}/trainer_{iter}.mp4", obsList, fps=16)
                bar.write(f"[video] trainer_{iter}.mp4 saved")
            except Exception:
                pass
            try:
                path = train.record_imagination(env, out_path=f"imagination_{it}.mp4",
                                                seed_steps=8, horizon=200, fps=20)
                print(f"[video] {path} saved")
            except Exception as e:
                print(f"[viz] skipped imagination: {e}")
        obsList = []
    else:
        
        buffer.add(obs, action.squeeze(0).cpu(), reward, not done)
        obs = next_obs
        prev_state = post_state
        prev_action = action
        score += reward
    bar.update(1)
print(np.mean(scores))