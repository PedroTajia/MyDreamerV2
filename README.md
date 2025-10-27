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
The repository ships with a simple training script and minimal components. A default run (with the included environment wrapper) looks like:
python run.py

Environment: task name, robot arm, reward shaping, frame size. (See env.py.) 

Model: RSSM sizes, stochastic/deterministic dims, encoder/decoder depth. (See rssm.py, image_codec.py.) 

Learning: batch size, sequence length, KL scale, actor/value learning rates. (See train.py, networks.py, actor.py.) 

Replay: capacity, sampling, episodic slicing. (See buffer.py.) 

If you use Weights & Biases:
`export WANDB_PROJECT=mydreamerv2`
python run.py
🧱 Repository Structure
MyDreamerV2/
├─ run.py            # Entry point for training/eval loops
├─ train.py          # World model + actor/value training steps
├─ env.py            # Gym wrapper for RoboSuite pixel observations
├─ image_codec.py    # CNN Encoder / Decoder for pixels
├─ rssm.py           # Recurrent State-Space Model (RSSM)
├─ networks.py       # Reward, Value, Discount heads
├─ actor.py          # Policy (actor) + action distribution
├─ buffer.py         # Replay buffer / episodic sampling

GitHub
🧠 Method (DreamerV2, in short)
DreamerV2 learns a discrete/structured latent world model (stochastic + deterministic states), optimizes reconstruction/reward/continuation losses with a KL regularizer, and then trains an actor–critic purely from imagined trajectories rolled out in latent space. This decouples representation learning from control while remaining sample-efficient. 

⚙️ Configuration Tips
Image size & channels: Keep encoder/decoder resolutions consistent across env.py and image_codec.py.
Sequence length: Long enough for credit assignment through imagination (e.g., 50–80), but balanced against memory.
KL balancing: If reconstructions look good but imagination is unstable, tune KL scale or free nats.
Action distribution: For continuous control, a squashed Gaussian (TanhNormal) is typical; ensure bounds match the env.

📊 Logging & Visualization
Metrics (losses, returns, KL, reconstruction MSE) and media (episode videos) can be logged to W&B.
For videos, stack frames as np.uint8 [T, H, W, 3] and log every N steps to avoid I/O slowdown.

