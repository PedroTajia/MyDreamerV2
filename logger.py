# logger.py
import os, re, datetime, dataclasses, sys
import numpy as np
import pandas as pd
from termcolor import colored
from logging import getLogger, Formatter, StreamHandler, INFO
from logging.handlers import RotatingFileHandler
import importlib

# ------- small helpers -------
def make_dir(p):
    try: os.makedirs(p, exist_ok=True)
    except OSError: pass
    return p

def _limstr(s, n=36):
    s = str(s)
    return (s[:n] + "...") if len(s) > n else s

def _obs_to_str(obs_shape):
    # supports list/tuple/dict (old code used dict.values())
    if isinstance(obs_shape, dict):
        vals = list(obs_shape.values())
    else:
        vals = list(obs_shape)
    return ", ".join(str(v) for v in vals)

def cfg_to_group(cfg, return_list=False):
    lst = [getattr(cfg, "task", "task"), re.sub("[^0-9a-zA-Z]+", "-", getattr(cfg, "exp_name", "exp"))]
    return lst if return_list else "-".join(lst)

# Console formatting presets (safe defaults)
CONSOLE_FORMAT = [
    ("step","step","int"),
    ("episode_reward","reward","float"),
    ("episode_success","success","float"),
    ("total_loss","total_loss","float"),  # was ("total_loss","loss","float")
    ("kl_loss","kl_loss","float"),
    ("obs_loss","obs_loss","float"),
    ("value_loss","value_loss","float"),
    ("actor_loss","actor_loss","float"),
    ("cont_loss", "cont_loss", "float"),
    ("reward_loss", "reward_loss", "float"),
    ("time","time","time"),
]
CAT_TO_COLOR = {"train": "green", "eval": "yellow", "pretrain": "cyan"}

# ------- video recorder -------
# video_recorder.py

class VideoRecorder:
    """
    Minimal W&B video logger when YOU provide frames.
    Expect frames as HxWxC (RGB) uint8. Converts if needed.
    """
    def __init__(self, wandb, fps=15):
        self._wandb = wandb
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def add(self, frame):
        """
        frame: np.ndarray
          - HxWxC (RGB) or CxHxW
          - any dtype; will be clipped->uint8
        """
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[0] < arr.shape[1]:
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self.frames.append(arr)

    def extend(self, frames):
        for f in frames:
            self.add(f)

    def save(self, step, key="videos/eval_video"):
        if not self._wandb or not self.frames:
            return
        T, H, W, C = len(self.frames), *self.frames[0].shape
        wb = self._wandb
        if not (wb and hasattr(wb, "log") and hasattr(wb, "Video")) or not self.frames:
            return
        frames = np.stack(self.frames)  # [T, H, W, C]
        wb.log(
            {key: wb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format="mp4")},
            step=step,
        )

# ------- main logger -------
class Logger:
    """Unified console/CSV/W&B logger (adapted to your code)."""

    def __init__(self, cfg):
        # paths & basic state
        self._log_dir = make_dir(cfg.work_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_csv = bool(getattr(cfg, "save_csv", False))
        self._save_agent = bool(getattr(cfg, "save_agent", False))
        self._group = cfg_to_group(cfg)
        self._seed = getattr(cfg, "seed", 0)
        self._eval_rows = []

        # python logging (console + rotating file)
        self._plog = self._build_pylogger(self._log_dir)

        # friendly run header
        self.print_run(cfg)

        # wandb gate from YAML (enable_wandb, wandb_project, wandb_silent)
        self.project = getattr(cfg, "wandb_project", "none")
        self.entity  = getattr(cfg, "wandb_entity", "none")
        self._wandb = None
        self._video = None
        if not getattr(cfg, "enable_wandb", False) or self.project in {"none", None} or self.entity in {"none", None}:
            self._plog.info("W&B disabled.")
            cfg.save_agent = False
            cfg.save_video = False
        else:
            os.environ["WANDB_SILENT"] = "true" if getattr(cfg, "wandb_silent", False) else "false"
            _wb = importlib.import_module("wandb")
            _wb.init(
                project=self.project,
                entity=self.entity,
                name=str(self._seed),
                group=self._group,
                tags=cfg_to_group(cfg, return_list=True) + [f"seed:{self._seed}"],
                dir=self._log_dir,
                config=dataclasses.is_dataclass(cfg) and dataclasses.asdict(cfg) or dict(cfg.__dict__),
            )
            self._plog.info("Logs will be synced with W&B.")
            self._wandb = _wb
            self._video = VideoRecorder(self._wandb, fps=getattr(cfg, "video_fps", 15)) if getattr(cfg, "save_video", False) else None


    # ---------- properties ----------
    @property
    def video(self): return self._video
    @property
    def model_dir(self): return self._model_dir

    # ---------- public API ----------
    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR, f"invalid category: {category}"
        # W&B
        if self._wandb:
            xkey = "step" if category in {"train", "eval"} else "iteration"
            self._wandb.log({f"{category}/{k}": v for k, v in d.items()}, step=d.get(xkey, 0))
        # CSV for eval
        if category == "eval" and self._save_csv and {"step", "episode_reward"} <= d.keys():
            self._eval_rows.append(np.array([d["step"], d["episode_reward"]]))
            pd.DataFrame(np.array(self._eval_rows)).to_csv(self._log_dir / "eval.csv",
                                                           header=["step", "episode_reward"], index=None)
        # console pretty print
        self._print(d, category)

    def save_agent(self, agent=None, identifier="final"):
        if self._save_agent and agent:
            fp = self._model_dir / f"{identifier}.pt"
            try:
                agent.save(fp)
                if self._wandb:
                    art = self._wandb.Artifact(self._group + f"-{self._seed}-{identifier}", type="model")
                    art.add_file(fp)
                    self._wandb.log_artifact(art)
                self._plog.info(f"Saved agent → {fp}")
            except Exception as e:
                self._plog.error(f"Failed to save agent: {e}")

    def finish(self, agent=None):
        try:
            self.save_agent(agent)
        except Exception as e:
            self._plog.error(f"Model save on finish failed: {e}")
        if self._wandb:
            self._wandb.finish()

    # ---------- internals ----------
    def _build_pylogger(self, log_dir):
        logger = getLogger("dreamer")
        if logger.handlers:  # prevent duplicates
            return logger
        logger.setLevel(INFO)
        fmt = Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        ch = StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(INFO)
        fh = RotatingFileHandler(os.path.join(str(log_dir), "train.log"), maxBytes=5_000_000, backupCount=2)
        fh.setFormatter(fmt); fh.setLevel(INFO)
        logger.addHandler(ch); logger.addHandler(fh); logger.propagate = False
        return logger

    def print_run(self, cfg):
        kvs = [
            ("steps", f"{int(getattr(cfg, 'steps', 0)):,}"),
            ("observations", _obs_to_str(getattr(cfg, 'obs_shape', [3,64,64]))),
            ("actions", getattr(cfg, "action_size", getattr(cfg, "action_dim", "?"))),
            ("experiment", getattr(cfg, "exp_name", "exp")),
        ]
        w = max(len(_limstr(v)) for _, v in kvs) + 25
        div = "-" * w
        print(div)
        for k, v in kvs:
            print("  " + colored(f"{k.capitalize()+':':<15}", "green", attrs=["bold"]), _limstr(v))
        print(div)
    # ---------- console pretty print ----------
    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key + ":", "blue")} {int(value):,}'
        if ty == "float":
            return f'{colored(key + ":", "blue")} {float(value):.01f}'
        if ty == "time":
            import datetime as _dt
            value = str(_dt.timedelta(seconds=int(value)))
            return f'{colored(key + ":", "blue")} {value}'
        return f'{colored(key + ":", "blue")} {value}'

    def _print(self, d, category):
        color = CAT_TO_COLOR.get(category, "white")
        pieces = [f' {colored(category, color):<14}']
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    