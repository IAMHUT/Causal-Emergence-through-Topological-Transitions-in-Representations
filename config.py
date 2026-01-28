# config.py
import torch
import json

class CFG:
    def __init__(self):
        self.env_name = "Pendulum-v1"
        self.obs_dim = 3
        self.action_dim = 1
        self.latent_dim = 32
        self.hidden_dim = 512
        self.num_domains = 5
        self.beta = 0.01
        self.lambda_e = 0.5
        self.gamma = 0.99
        self.tau = 0.005
        self.exploration_noise = 0.5
        self.exploration_steps = 50000
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.batch_size = 256
        self.buffer_size = 200000
        self.total_steps = 450000
        self.warmup_steps = 10000
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.lr_world_model = 1e-4
        self.lr_encoder = 1e-4
        self.lr_discriminator = 1e-4
        self.alpha = 0.2
        self.target_entropy = -torch.prod(torch.Tensor([1])).item()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_freq = 1
        self.crd_update_freq = 2
        self.sac_update_freq = 1
        self.log_interval = 1000
        self.eval_interval = 5000
        self.save_interval = 50000
        self.min_exploration_noise = 0.1
        self.adaptive_exploration = True

    def to_dict(self):
        cfg_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.device):
                cfg_dict[k] = str(v)
            else:
                cfg_dict[k] = v
        return cfg_dict

    @classmethod
    def from_dict(cls, cfg_dict):
        cfg = cls()
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                if k == 'device' and isinstance(v, str):
                    setattr(cfg, k, torch.device(v))
                else:
                    setattr(cfg, k, v)
        return cfg

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            cfg_dict = json.load(f)
        return cls.from_dict(cfg_dict)