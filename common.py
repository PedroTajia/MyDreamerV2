import yaml
import numpy as np
import wandb

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
