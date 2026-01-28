# crd_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRD:
    def __init__(self, beta=0.03, lambda_e=0.3):
        self.beta = beta
        self.lambda_e = lambda_e

    def compute_rate(self, mu, logvar):
        kl = 0.5 * torch.sum(
            torch.exp(logvar) + mu.pow(2) - 1 - logvar,
            dim=-1
        )
        return kl.mean()

    def compute_distortion(self, pred_z_mu, pred_z_logvar, target_z, pred_reward, target_reward):
        log_var = pred_z_logvar
        var = torch.exp(log_var)
        nll_state = 0.5 * torch.sum(
            log_var + (target_z - pred_z_mu).pow(2) / (var + 1e-8),
            dim=-1
        )
        nll_state = nll_state.mean()
        reward_loss = F.mse_loss(pred_reward, target_reward)
        distortion = nll_state + 0.5 * reward_loss
        return distortion, nll_state.item(), reward_loss.item()

    def compute_adversarial_loss(self, discriminator, z, domain_labels):
        domain_logits = discriminator(z, adversarial=True)
        adversarial_loss = F.cross_entropy(domain_logits, domain_labels)
        return adversarial_loss

    def compute_encoder_world_model_loss(self, encoder, world_model, discriminator,
                                         obs, action, next_obs, reward, domain_labels):
        z, mu, logvar = encoder.sample(obs)
        with torch.no_grad():
            target_z, _, _ = encoder.sample(next_obs)
        pred_z_mu, pred_z_logvar, pred_reward = world_model(z, action)
        distortion, nll_state, nll_reward = self.compute_distortion(
            pred_z_mu, pred_z_logvar, target_z, pred_reward, reward
        )
        rate = self.compute_rate(mu, logvar)
        adversarial_loss = self.compute_adversarial_loss(discriminator, z, domain_labels)
        total_loss = distortion + self.beta * rate - self.lambda_e * adversarial_loss
        return {
            'loss': total_loss,
            'distortion': distortion.item(),
            'rate': rate.item(),
            'adversarial_loss': adversarial_loss.item(),
            'nll_state': nll_state,
            'nll_reward': nll_reward
        }

    def compute_discriminator_loss(self, discriminator, z, domain_labels):
        domain_logits = discriminator(z, adversarial=False)
        loss = F.cross_entropy(domain_logits, domain_labels)
        return loss