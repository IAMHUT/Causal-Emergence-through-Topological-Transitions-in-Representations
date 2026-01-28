# environment.py
import gym
import numpy as np

class DomainShiftWrapper(gym.Wrapper):
    def __init__(self, env, shift_type='none', strength=0.0):
        super().__init__(env)
        self.shift_type = shift_type
        self.strength = strength
        self.original_params = self._save_original_params()
        self._apply_shift()
        self._gym_new_api = hasattr(self.env, 'reset') and callable(getattr(self.env.reset, '__call__', None))

    def _save_original_params(self):
        params = {}
        if hasattr(self.env.unwrapped, 'm'):
            params['mass'] = self.env.unwrapped.m
        if hasattr(self.env.unwrapped, 'l'):
            params['length'] = self.env.unwrapped.l
        return params

    def _apply_shift(self):
        if self.shift_type == 'none':
            return
        elif self.shift_type == 'visual':
            self.observation_noise = self.strength * 0.1
        elif self.shift_type == 'dynamics':
            if hasattr(self.env.unwrapped, 'm'):
                self.env.unwrapped.m = self.original_params.get('mass', 1.0) * (1.0 + self.strength * 0.5)
            if hasattr(self.env.unwrapped, 'l'):
                self.env.unwrapped.l = self.original_params.get('length', 1.0) * (1.0 + self.strength * 0.3)
        elif self.shift_type == 'intervention':
            self.action_scale = 1.0 - self.strength * 0.5
            self.action_delay = max(1, int(self.strength * 3))
            self.action_buffer = []
        elif self.shift_type == 'non_mechanism':
            self.reward_shift = self.strength * 0.5

    def reset(self, **kwargs):
        try:
            obs, info = self.env.reset(**kwargs)
            info = info if info is not None else {}
        except:
            obs = self.env.reset()
            info = {}
        if self.shift_type == 'visual' and hasattr(self, 'observation_noise'):
            obs = obs + np.random.normal(0, self.observation_noise, obs.shape)
        if self.shift_type == 'intervention':
            self.action_buffer = []
        info['domain_id'] = self._get_domain_id()
        return obs, info

    def step(self, action):
        if self.shift_type == 'intervention':
            if hasattr(self, 'action_scale'):
                action = action * self.action_scale
            if hasattr(self, 'action_delay'):
                self.action_buffer.append(action)
                if len(self.action_buffer) > self.action_delay:
                    action = self.action_buffer.pop(0)
                else:
                    action = np.zeros_like(action)
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        except:
            obs, reward, done, info = self.env.step(action)
            terminated = done
            truncated = False
        if self.shift_type == 'visual' and hasattr(self, 'observation_noise'):
            obs = obs + np.random.normal(0, self.observation_noise, obs.shape)
        if self.shift_type == 'non_mechanism' and hasattr(self, 'reward_shift'):
            reward = reward * (1.0 - self.reward_shift)
        domain_id = self._get_domain_id()
        info['domain_id'] = domain_id
        if self._gym_new_api:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, done, info

    def _get_domain_id(self):
        domain_map = {
            'none': 0,
            'visual': 1,
            'dynamics': 2,
            'intervention': 3,
            'non_mechanism': 4
        }
        return domain_map.get(self.shift_type, 0)

class DomainCurriculum:
    def __init__(self, shift_types, max_strength=0.5, steps_per_stage=50000):
        self.shift_types = shift_types
        self.max_strength = max_strength
        self.steps_per_stage = steps_per_stage
        self.current_stage = 0
        self.total_steps = 0

    def update(self, steps):
        self.total_steps += steps
        self.current_stage = min(
            len(self.shift_types) - 1,
            self.total_steps // self.steps_per_stage
        )

    def get_current_domain(self):
        if self.current_stage >= len(self.shift_types):
            return self.shift_types[-1]
        shift_type = self.shift_types[self.current_stage]
        progress = (self.total_steps % self.steps_per_stage) / self.steps_per_stage
        strength = min(self.max_strength, progress * self.max_strength)
        return shift_type, strength

    def get_all_domains(self):
        domains = []
        for shift_type in self.shift_types:
            for strength in [0.0, 0.3, 0.6, 0.9]:
                domains.append((shift_type, strength))
        return domains