# replay_buffer.py
import numpy as np
import random
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'domain'])

class CRDReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim, num_domains=3):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.num_domains = num_domains
        self.buffer = deque(maxlen=capacity)
        self.domain_buffers = [deque(maxlen=capacity // num_domains) for _ in range(num_domains)]
        self.position = 0
        self.domain_counts = [0] * num_domains

    def add(self, state, action, reward, next_state, done, domain=None):
        if domain is None:
            domain = np.random.randint(0, self.num_domains)
        experience = Experience(state, action, reward, next_state, done, domain)
        self.buffer.append(experience)
        self.domain_buffers[domain].append(experience)
        self.domain_counts[domain] = min(self.domain_counts[domain] + 1, self.capacity // self.num_domains)

    def sample(self, batch_size, balanced_domains=False):
        if len(self.buffer) < batch_size:
            return None
        if balanced_domains:
            samples_per_domain = max(1, batch_size // self.num_domains)
            batch = []
            for domain in range(self.num_domains):
                if len(self.domain_buffers[domain]) >= samples_per_domain:
                    domain_samples = random.sample(self.domain_buffers[domain], samples_per_domain)
                    batch.extend(domain_samples)
            if len(batch) < batch_size:
                additional = random.sample(self.buffer, batch_size - len(batch))
                batch.extend(additional)
            batch = batch[:batch_size]
        else:
            batch = random.sample(self.buffer, batch_size)
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        domains = np.array([exp.domain for exp in batch])
        return states, actions, rewards, next_states, dones, domains

    def sample_from_domain(self, domain, batch_size):
        if len(self.domain_buffers[domain]) < batch_size:
            return None
        batch = random.sample(self.domain_buffers[domain], batch_size)
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        domains = np.array([exp.domain for exp in batch])
        return states, actions, rewards, next_states, dones, domains

    def __len__(self):
        return len(self.buffer)

    def get_domain_statistics(self):
        return {
            'total_samples': len(self.buffer),
            'domain_counts': self.domain_counts,
            'domain_ratios': [c / max(sum(self.domain_counts), 1) for c in self.domain_counts]
        }