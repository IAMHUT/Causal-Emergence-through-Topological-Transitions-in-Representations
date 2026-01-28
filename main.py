# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import time
import argparse
from collections import defaultdict
import os
import json
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from config import CFG
from networks import CausalEncoder, WorldModel, DomainDiscriminator, GaussianPolicy, TwinQNetwork
from crd_loss import CRD
from replay_buffer import CRDReplayBuffer
from environment import DomainShiftWrapper, DomainCurriculum

class CRDANRAgent:
    def __init__(self, config):
        self.cfg = config
        self.device = config.device
        self._init_networks()
        self._init_optimizers()
        self._init_loss()
        self.buffer = CRDReplayBuffer(
            config.buffer_size,
            (config.obs_dim,),
            config.action_dim,
            config.num_domains
        )
        self.total_steps = 0
        self.episode_rewards = []
        self.metrics_history = defaultdict(list)
        self.exploration_noise_schedule = config.exploration_noise
        self.domain_curriculum = DomainCurriculum(
            shift_types=['none', 'visual', 'dynamics', 'intervention'],
            max_strength=0.5,
            steps_per_stage=50000
        )
        self.best_reward = -float('inf')
        self.patience_counter = 0

    def _init_networks(self):
        self.encoder = CausalEncoder(
            self.cfg.obs_dim,
            self.cfg.latent_dim,
            self.cfg.hidden_dim
        ).to(self.device)
        self.world_model = WorldModel(
            self.cfg.latent_dim,
            self.cfg.action_dim,
            self.cfg.hidden_dim
        ).to(self.device)
        self.discriminator = DomainDiscriminator(
            self.cfg.latent_dim,
            self.cfg.num_domains,
            self.cfg.hidden_dim // 2
        ).to(self.device)
        self.actor = GaussianPolicy(
            self.cfg.latent_dim,
            self.cfg.action_dim,
            self.cfg.hidden_dim
        ).to(self.device)
        self.critic = TwinQNetwork(
            self.cfg.latent_dim,
            self.cfg.action_dim,
            self.cfg.hidden_dim
        ).to(self.device)
        self.critic_target = TwinQNetwork(
            self.cfg.latent_dim,
            self.cfg.action_dim,
            self.cfg.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _init_optimizers(self):
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.lr_encoder,
            weight_decay=1e-5
        )
        self.world_model_optimizer = optim.Adam(
            self.world_model.parameters(),
            lr=self.cfg.lr_world_model,
            weight_decay=1e-5
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.lr_discriminator,
            weight_decay=1e-5
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.cfg.lr_actor,
            weight_decay=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.cfg.lr_critic,
            weight_decay=1e-5
        )

    def _init_loss(self):
        self.crd_loss = CRD(
            beta=self.cfg.beta,
            lambda_e=self.cfg.lambda_e
        )

    def select_action(self, state, deterministic=False, exploration=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z, _, _ = self.encoder.sample(state, deterministic=True)
        if exploration and not deterministic:
            if self.total_steps < self.cfg.exploration_steps:
                current_noise = self.cfg.exploration_noise * max(0.2, 1.0 - self.total_steps / self.cfg.exploration_steps)
                if np.random.random() < 0.3:
                    action = np.random.uniform(-1, 1, self.cfg.action_dim)
                else:
                    with torch.no_grad():
                        action, _, _, _ = self.actor.sample(z, deterministic=False, with_logprob=False)
                    action = action.cpu().numpy()[0]
                    noise = np.random.normal(0, current_noise, self.cfg.action_dim)
                    action = np.clip(action + noise, -1, 1)
            else:
                with torch.no_grad():
                    action, _, _, _ = self.actor.sample(z, deterministic=False, with_logprob=False)
                action = action.cpu().numpy()[0]
                noise_scale = max(self.cfg.min_exploration_noise, 0.4 * np.exp(-self.total_steps / 50000))
                noise = np.random.normal(0, noise_scale, self.cfg.action_dim)
                action = np.clip(action + noise, -1, 1)
        else:
            with torch.no_grad():
                action, _, _, _ = self.actor.sample(z, deterministic=True, with_logprob=False)
            action = action.cpu().numpy()[0]
        return action

    def update_crd(self, batch):
        states, actions, rewards, next_states, dones, domains = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        domains = torch.LongTensor(domains).to(self.device)
        with torch.no_grad():
            z, _, _ = self.encoder.sample(states)
        discriminator_loss = self.crd_loss.compute_discriminator_loss(self.discriminator, z, domains)
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        self.discriminator_optimizer.step()
        crd_metrics = self.crd_loss.compute_encoder_world_model_loss(
            self.encoder, self.world_model, self.discriminator,
            states, actions, next_states, rewards, domains
        )
        self.encoder_optimizer.zero_grad()
        self.world_model_optimizer.zero_grad()
        crd_metrics['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 0.5)
        self.encoder_optimizer.step()
        self.world_model_optimizer.step()
        return {
            'crd_loss': crd_metrics['loss'].item(),
            'distortion': crd_metrics['distortion'],
            'rate': crd_metrics['rate'],
            'adversarial_loss': crd_metrics['adversarial_loss'],
            'discriminator_loss': discriminator_loss.item()
        }

    def update_sac(self, batch):
        states, actions, rewards, next_states, dones, domains = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            z, _, _ = self.encoder.sample(states, deterministic=True)
            next_z, _, _ = self.encoder.sample(next_states, deterministic=True)
            next_action, next_log_prob, _, _ = self.actor.sample(next_z, deterministic=False, with_logprob=True)
            target_q1, target_q2 = self.critic_target(next_z, next_action)
            target_q = torch.min(target_q1, target_q2) - self.cfg.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.cfg.gamma * target_q
        current_q1, current_q2 = self.critic(z, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        for param in self.critic.parameters():
            param.requires_grad = False
        new_action, log_prob, _, _ = self.actor.sample(z, deterministic=False, with_logprob=True)
        q1_new, q2_new = self.critic(z, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.cfg.alpha * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        for param in self.critic.parameters():
            param.requires_grad = True
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
            )
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q1.mean().item(),
            'log_prob': log_prob.mean().item() if log_prob is not None else 0
        }

    def train_episode(self, env):
        try:
            state, info = env.reset()
            info = info if info is not None else {}
        except:
            state = env.reset()
            info = {}
        episode_reward = 0
        episode_steps = 0
        done = False
        terminated = False
        truncated = False
        while not (done or terminated or truncated) and episode_steps < 200:
            action = self.select_action(state, exploration=True)
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                info = info if info is not None else {}
            except:
                next_state, reward, done, info = env.step(action)
                terminated = done
                truncated = False
                info = info if info is not None else {}
            domain_id = info.get('domain_id', 0)
            self.buffer.add(state, action, reward, next_state, done, domain_id)
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            if self.total_steps > self.cfg.warmup_steps:
                batch = self.buffer.sample(self.cfg.batch_size, balanced_domains=True)
                if batch is not None:
                    if self.total_steps % self.cfg.crd_update_freq == 0:
                        crd_metrics = self.update_crd(batch)
                        if self.total_steps % 500 == 0:
                            for key, value in crd_metrics.items():
                                self.metrics_history[key].append(value)
                    if self.total_steps % self.cfg.sac_update_freq == 0:
                        sac_metrics = self.update_sac(batch)
                        if self.total_steps % 500 == 0:
                            for key, value in sac_metrics.items():
                                self.metrics_history[key].append(value)
            if self.total_steps % 1000 == 0 and self.cfg.adaptive_exploration:
                self.domain_curriculum.update(1000)
                progress = min(1.0, self.total_steps / self.cfg.exploration_steps)
                self.exploration_noise_schedule = max(
                    self.cfg.min_exploration_noise,
                    0.5 * np.exp(-3 * progress)
                )
            state = next_state
        return episode_reward, episode_steps

    def evaluate(self, env, num_episodes=3):
        total_rewards = []
        for _ in range(num_episodes):
            try:
                state, info = env.reset()
                info = info if info is not None else {}
            except:
                state = env.reset()
                info = {}
            episode_reward = 0
            done = False
            terminated = False
            truncated = False
            episode_steps = 0
            while not (done or terminated or truncated) and episode_steps < 200:
                action = self.select_action(state, deterministic=True, exploration=False)
                try:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                except:
                    next_state, reward, done, info = env.step(action)
                    terminated = done
                    truncated = False
                episode_reward += reward
                state = next_state
                episode_steps += 1
            total_rewards.append(episode_reward)
        return np.mean(total_rewards), np.std(total_rewards)

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable_metrics = {}
        for key, values in self.metrics_history.items():
            serializable_metrics[key] = list(values)
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'world_model': self.world_model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'optimizers': {
                'encoder': self.encoder_optimizer.state_dict(),
                'world_model': self.world_model_optimizer.state_dict(),
                'discriminator': self.discriminator_optimizer.state_dict(),
                'actor': self.actor_optimizer.state_dict(),
                'critic': self.critic_optimizer.state_dict(),
            },
            'total_steps': self.total_steps,
            'metrics_history': serializable_metrics,
            'episode_rewards': self.episode_rewards,
            'config_dict': self.cfg.to_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.encoder_optimizer.load_state_dict(checkpoint['optimizers']['encoder'])
        self.world_model_optimizer.load_state_dict(checkpoint['optimizers']['world_model'])
        self.discriminator_optimizer.load_state_dict(checkpoint['optimizers']['discriminator'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizers']['actor'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizers']['critic'])
        self.total_steps = checkpoint['total_steps']
        self.metrics_history = defaultdict(list, checkpoint['metrics_history'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        config_dict = checkpoint['config_dict']
        self.cfg = CFG.from_dict(config_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=300000)
    parser.add_argument('--save_dir', type=str, default='./crd_anr_results_improved')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--lambda_e', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--exploration_steps', type=int, default=50000)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env)
    try:
        obs, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
    except:
        try:
            env.seed(args.seed)
        except:
            pass
    config = CFG()
    config.env_name = args.env
    config.total_steps = args.total_steps
    config.beta = args.beta
    config.lambda_e = args.lambda_e
    config.batch_size = args.batch_size
    config.exploration_steps = args.exploration_steps
    agent = CRDANRAgent(config)
    if args.load_checkpoint:
        agent.load_checkpoint(args.load_checkpoint)
    os.makedirs(args.save_dir, exist_ok=True)
    config.save(f"{args.save_dir}/config.json")
    print("Starting optimized CRD-ANR training...")
    start_time = time.time()
    episode = 0
    best_reward = -float('inf')
    patience_counter = 0
    patience = 20
    try:
        while agent.total_steps < config.total_steps:
            episode_reward, episode_steps = agent.train_episode(env)
            agent.episode_rewards.append(episode_reward)
            episode += 1
            if episode_reward > best_reward:
                best_reward = episode_reward
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience and episode > 50 and config.adaptive_exploration:
                agent.exploration_noise_schedule = min(0.3, agent.exploration_noise_schedule * 1.5)
                patience_counter = 0
            if episode % 10 == 0:
                avg_reward = np.mean(agent.episode_rewards[-10:]) if agent.episode_rewards else 0
                elapsed_time = time.time() - start_time
                steps_per_sec = agent.total_steps / elapsed_time if elapsed_time > 0 else 0
                print(f"Episode {episode:4d} | Total steps {agent.total_steps:7d} | Episode reward {episode_reward:7.2f} | Avg reward {avg_reward:7.2f} | Best reward {best_reward:7.2f} | Steps/sec {steps_per_sec:5.1f} | Exploration {agent.exploration_noise_schedule:.3f}")
            if episode % 25 == 0 and episode > 0:
                test_envs = {
                    'id': DomainShiftWrapper(env, shift_type='none'),
                    'visual_mild': DomainShiftWrapper(env, shift_type='visual', strength=0.3),
                    'visual_severe': DomainShiftWrapper(env, shift_type='visual', strength=0.6),
                }
                for env_name, test_env in test_envs.items():
                    mean_reward, std_reward = agent.evaluate(test_env, num_episodes=2)
                    print(f"  {env_name:15s}: {mean_reward:7.2f} ± {std_reward:5.2f}")
            if agent.total_steps % config.save_interval == 0 and agent.total_steps > 0:
                checkpoint_path = f"{args.save_dir}/crd_anr_checkpoint_{agent.total_steps}.pth"
                agent.save_checkpoint(checkpoint_path)
                reward_history = {
                    'episode_rewards': agent.episode_rewards,
                    'episode': episode,
                    'total_steps': agent.total_steps
                }
                with open(f"{args.save_dir}/reward_history.json", 'w') as f:
                    json.dump(reward_history, f)
    except KeyboardInterrupt:
        pass
    test_envs = {
        'id': DomainShiftWrapper(env, shift_type='none'),
        'visual_mild': DomainShiftWrapper(env, shift_type='visual', strength=0.3),
        'visual_severe': DomainShiftWrapper(env, shift_type='visual', strength=0.6),
        'dynamics_mild': DomainShiftWrapper(env, shift_type='dynamics', strength=0.2),
        'dynamics_severe': DomainShiftWrapper(env, shift_type='dynamics', strength=0.4),
    }
    for env_name, test_env in test_envs.items():
        mean_reward, std_reward = agent.evaluate(test_env, num_episodes=3)
        print(f"{env_name:15s}: {mean_reward:7.2f} ± {std_reward:5.2f}")
    final_path = f"{args.save_dir}/crd_anr_final.pth"
    agent.save_checkpoint(final_path)
    plot_training_results(agent, args.save_dir)
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    if agent.episode_rewards:
        last_100_avg = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
        print(f"Average reward (last 100 episodes): {last_100_avg:.2f}")

def plot_training_results(agent, save_dir):
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        if agent.episode_rewards:
            axes[0, 0].plot(agent.episode_rewards, alpha=0.6, linewidth=1, color='blue')
            window_size = min(20, len(agent.episode_rewards) // 10)
            if window_size > 1:
                moving_avg = np.convolve(agent.episode_rewards, np.ones(window_size) / window_size, mode='valid')
                axes[0, 0].plot(range(window_size - 1, len(agent.episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (n={window_size})')
                axes[0, 0].legend()
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        if 'crd_loss' in agent.metrics_history and agent.metrics_history['crd_loss']:
            axes[0, 1].plot(agent.metrics_history['crd_loss'], color='purple')
            axes[0, 1].set_title('CRD Loss')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        if 'rate' in agent.metrics_history and 'distortion' in agent.metrics_history:
            if agent.metrics_history['rate'] and agent.metrics_history['distortion']:
                min_len = min(len(agent.metrics_history['rate']), len(agent.metrics_history['distortion']))
                axes[0, 2].plot(agent.metrics_history['rate'][:min_len], label='Rate', color='blue')
                axes[0, 2].plot(agent.metrics_history['distortion'][:min_len], label='Distortion', color='red')
                axes[0, 2].set_title('Rate vs Distortion')
                axes[0, 2].set_xlabel('Training Steps')
                axes[0, 2].set_ylabel('Value')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
        if 'critic_loss' in agent.metrics_history and 'actor_loss' in agent.metrics_history:
            if agent.metrics_history['critic_loss'] and agent.metrics_history['actor_loss']:
                min_len = min(len(agent.metrics_history['critic_loss']), len(agent.metrics_history['actor_loss']))
                axes[1, 0].plot(agent.metrics_history['critic_loss'][:min_len], label='Critic', color='blue')
                axes[1, 0].plot(agent.metrics_history['actor_loss'][:min_len], label='Actor', color='red')
                axes[1, 0].set_title('SAC Losses')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        if 'q_value' in agent.metrics_history and agent.metrics_history['q_value']:
            axes[1, 1].plot(agent.metrics_history['q_value'], color='green')
            axes[1, 1].set_title('Q Value')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].grid(True, alpha=0.3)
        if 'adversarial_loss' in agent.metrics_history and agent.metrics_history['adversarial_loss']:
            axes[1, 2].plot(agent.metrics_history['adversarial_loss'], color='orange')
            axes[1, 2].set_title('Adversarial Loss')
            axes[1, 2].set_xlabel('Training Steps')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True, alpha=0.3)
        plt.suptitle('CRD-ANR Training Diagnostics')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/detailed_training_curves.png", dpi=150, bbox_inches='tight')
        plt.figure(figsize=(12, 6))
        if agent.episode_rewards:
            plt.plot(agent.episode_rewards, alpha=0.6, linewidth=1, label='Raw Reward', color='blue')
            window_size = min(50, len(agent.episode_rewards) // 5)
            if window_size > 1:
                moving_avg = np.convolve(agent.episode_rewards, np.ones(window_size) / window_size, mode='valid')
                plt.plot(range(window_size - 1, len(agent.episode_rewards)), moving_avg, 'r-', linewidth=3, label=f'Moving Avg (n={window_size})')
            plt.title('CRD-ANR Training Reward Curve')
            plt.xlabel('Episode')
            plt.ylabel('Total Episode Reward')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.savefig(f"{save_dir}/reward_curve.png", dpi=150, bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        pass

if __name__ == "__main__":
    main()