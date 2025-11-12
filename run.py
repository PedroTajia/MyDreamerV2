import wandb
import torch
from env import RobotSuiteEnv
from common import load_yaml_dict, DotDict
from train import Train
from pathlib import Path
from datetime import datetime
from logger import Logger



# usage

device = "mps" if torch.backends.mps.is_available( ) else "cpu"
device = "cuda" if torch.cuda.is_available() else device
print(f"Device: {device}")


def main():
    info = load_yaml_dict("config.yaml") 
    
    name = info.get("exp_name") or f"{info.get('task','exp')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    seed = info.get("seed", 0)
    info.update({
        "exp_name": name,
        "seed": seed,
        "work_dir": Path("./runs") / name / str(seed),
        "steps": int(info.get("steps", 0)),
    })
    
    info = DotDict(info)
    
    log = Logger(info)
    
    train = Train(info, device=device)
    
    env = RobotSuiteEnv(env_name='Lift', robot="Panda", output_obs=info['obs_shape'])

    prev_state = train.rssm._init_rssm(1)
    prev_action = torch.tensor(env.action_space.sample()).unsqueeze(0).to(device=device)
    
    done = False
    
    obs, score = env.reset(), 0

    
    video = log.video  
    
    done_num = 0
    
    t_in_ep = 0

    
    try:
        for iter in range( int(info['steps'])):
            # training last
            
    
            if iter > info['seed'] and iter%info['train_every'] == 0:
                total_loss, kl_loss, reward_loss, cont_loss, prior_dist, post_dist, obs_loss, perpix_mse, actor_loss, value_loss, explore_loss = train.train_batch()
                metrics = {     
                                "step": iter,
                                "total_loss": float(total_loss.detach()),
                                "reward_loss": float(reward_loss.detach()),
                                "cont_loss": float(cont_loss.detach()),
                                "obs_loss": float(obs_loss.detach()),
                                "actor_loss": float(actor_loss.detach()),
                                "value_loss": float(value_loss.detach()),
                                "kl_loss": float(kl_loss.detach()),
                                "episode_reward": score
                                }
            
                if info["explore_only"]:
                    metrics["explore_loss"] = explore_loss
                
                log.log(metrics, category="train")

            if iter%info['target_update'] == 0:
                    train.update_target()
                

                
            with torch.no_grad():
                obs = obs.to(device, dtype=torch.float32) 
                embed = train.encoder(obs).unsqueeze(0).to(device)
                
                
                done_t = torch.as_tensor(done, dtype=torch.bool, device=device).view(1, 1)
                cont = (~done_t).to(torch.float32)
                
                
                _, post_state = train.rssm.rssm_obs(embed, prev_action, cont , prev_state)
                model_state = train.rssm.get_state(post_state).to(device)
                action, action_dist = train.actorModel(model_state)
                action = action.detach()
                
                video.add(env.render())
                if t_in_ep == 0:
                    rssm_first_post = post_state

            next_obs, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
            
            if done:
                done_num += 1
                
                if done_num%25== 0:
                    if len(video.frames) > 256:
                        video.frames = video.frames[-256:]
                        
                    video.save(iter, key = "video/rollout")
                    video.reset()
                    
                if done_num%10== 0:    
                    with torch.no_grad():
                        video.reset()
                        train.rssm.eval(); train.decoder.eval()
                        
                        start_state = rssm_first_post if rssm_first_post is not None else prev_state

                        imag_state, _, _ = train.rssm.rollout_imag(info['horizon'] , train.actorModel, start_state)
                                
                        imag_modelstate = train.rssm.get_state(imag_state)
                        obs_dist = train.decoder(imag_modelstate)
                        obs_mean = getattr(obs_dist, "mean")  # [T, B, C, H, W]

                        for t in range(len(obs_mean)):
                            frame = (obs_mean[t, 0].clamp(0, 1) * 255).to(torch.uint8)
                            frame = frame.permute(1, 2, 0).cpu().numpy() 
                            video.add(frame)
                        video.save(iter, key = "video/rollout_rssm")
                        video.reset()            
                
                if info["explore_only"]:
                    intrinsic = train.plan2explore._intrinsic_reward(imag_modelstate) 
                    log.log({"step": iter, "intrinsic_reward":intrinsic.mean(), }, category="train")
                    
                
                train.buffer.add(next_obs, action.squeeze(0).cpu(), reward, done)
                obs, score = env.reset(), 0
                prev_state = train.rssm._init_rssm(1)
                
                rssm_first_post = None
                t_in_ep = 0


                prev_action = torch.tensor(env.action_space.sample()).unsqueeze(0).to(device=device)
                done = False
                continue
                            
                        
                
                
            else:
                train.buffer.add(obs, action.squeeze(0).cpu(), reward, done)
                obs = next_obs
                prev_state = post_state
                prev_action = action
                score += reward
                t_in_ep += 1
                
            
            
            

                    
    finally:
            # Make shutdown orderly so no semaphores remain registered
            try:
                env.close()
            except Exception:
                pass
            try:
                wandb.finish() 
            except Exception:
                pass
    
if __name__ == "__main__":
  
    main()