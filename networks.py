# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)

class CausalEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -10, 2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, x, deterministic=False):
        mu, logvar = self.forward(x)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class WorldModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        dynamics_out = self.dynamics_net(x)
        z_next_mu, z_next_logvar = torch.chunk(dynamics_out, 2, dim=-1)
        z_next_logvar = torch.clamp(z_next_logvar, -10, 2)
        reward = self.reward_net(x)
        return z_next_mu, z_next_logvar, reward

class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_domains, hidden_dim=128):
        super().__init__()
        self.num_domains = num_domains
        self.grl = GradientReversalLayer(alpha=1.0)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_domains),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, z, adversarial=False):
        if adversarial:
            z = self.grl(z)
        return self.net(z)

class GaussianPolicy(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 2, action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is self.mu_head:
                nn.init.uniform_(module.weight, -3e-3, 3e-3)
            else:
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, z):
        h = self.net(z)
        mu = torch.tanh(self.mu_head(h))
        log_std = torch.clamp(self.log_std_head(h), -20, 2)
        return mu, log_std

    def sample(self, z, deterministic=False, with_logprob=True):
        mu, log_std = self.forward(z)
        if deterministic:
            action = mu
            log_prob = None
        else:
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mu, std)
            action_raw = normal.rsample()
            action = torch.tanh(action_raw)
            if with_logprob:
                log_prob = normal.log_prob(action_raw)
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(-1, keepdim=True)
            else:
                log_prob = None
        return action, log_prob, mu, log_std

class TwinQNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2