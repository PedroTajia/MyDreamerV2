import yaml
import numpy as np
import wandb
import torch
import re
def load_yaml_dict(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

# usage
cfg = load_yaml_dict() 

class DotDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v
    def __delattr__(self, k):
        del self[k]

def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)

def one_state(state):
    return (state[0][0], state[1][0], state[2][0])