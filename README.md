MyDreamerV2
A PyTorch re-implementation of DreamerV2 for pixel-based control.
Built around an RSSM world model, actor–critic learning in imagination, and practical tooling for training and logging. The repo currently includes minimal components (RSSM, encoder/decoder, reward/value/discount models, actor, replay buffer, and a RoboSuite gym wrapper).

## ✨ Highlights
PyTorch-first code with small, readable modules (RSSM, encoders/decoders, actor, buffers). 
GitHub
Pixel-based continuous control setup (e.g., RoboSuite Lift) via a custom Gym wrapper. 
GitHub
End-to-end world model training (reconstruction, reward, continuation) + imagined rollouts for policy/value learning, following DreamerV2. 

## 📦 Installation
Tested with Python 3.10+ and PyTorch. If you’re on macOS (MPS) or CUDA, install the appropriate PyTorch build from pytorch.org first.


# 1) Install core deps
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
`pip install gymnasium pygame imageio tqdm numpy matplotlib pillow`

# 2) RoboSuite (for pixel control tasks)
`pip install robosuite==1.4.1`

# 3) (Optional) experiment tracking
`pip install wandb`
DreamerV2 background: paper and project page for conceptual details. 

🚀 Quick Start
The repository ships with a simple training script and minimal components. 

Environment: task name, robot arm, reward shaping, frame size. (See env.py.) 

Model: RSSM sizes, stochastic/deterministic dims, encoder/decoder depth. (See rssm.py, image_codec.py.) 

Learning: batch size, sequence length, KL scale, actor/value learning rates. (See train.py, networks.py, actor.py.) 

Replay: capacity, sampling, episodic slicing. (See buffer.py.) 

If you use Weights & Biases:
`export WANDB_PROJECT=mydreamerv2`
python run.py


## 📁 Repository Structure

```text
MyDreamerV2/
├─ run.py            # Entry point for training/eval loops
├─ train.py          # World model + actor/value training steps
├─ env.py            # Gym wrapper for RoboSuite pixel observations
├─ image_codec.py    # CNN Encoder / Decoder for pixels
├─ rssm.py           # Recurrent State-Space Model (RSSM)
├─ networks.py       # Reward, Value, Discount heads
├─ actor.py          # Policy (actor) + action distribution
├─ buffer.py         # Replay buffer / episodic sampling
```

## 🧠 Method (DreamerV2, in short)
DreamerV2 learns a discrete/structured latent world model (stochastic + deterministic states), optimizes reconstruction/reward/continuation losses with a KL regularizer, and then trains an actor–critic purely from imagined trajectories rolled out in latent space. 

## 📊 Training Performance

Below is the **training episode return** over environment steps for a pixel-based RoboSuite task.

![Training episode return](assets/episode_return.png)

### Interpreting the plot

- **0 – ~50k steps**  
  Near-zero returns. The agent mainly explores randomly while the world model is inaccurate.

- **~50k – ~150k steps**  
  Gradual improvement as the RSSM learns meaningful latent dynamics and imagined rollouts become informative.

- **~150k – ~220k steps**  
  Clear learning signal. Episode returns increase consistently, indicating effective policy learning.

- **~220k+ steps (success regime)**  
  The agent frequently completes the task. While variance remains high (typical for pixel-based control), the mean return stays well above early-training levels.

**Task success begins around ~180k–200k environment steps**, where returns become consistently positive and structured rather than sparse spikes.
