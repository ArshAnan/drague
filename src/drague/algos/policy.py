"""Shared actor-critic. Mean-pooling other drones makes the weights independent of team size."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

_ORTHO = nn.initializers.orthogonal


class ActorCritic(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, self_feat, others):
        # self_feat (..., 10), others (..., M, 10). Leading batch axes are free.
        s = nn.tanh(nn.Dense(self.hidden, kernel_init=_ORTHO(jnp.sqrt(2.0)))(self_feat))
        o = nn.tanh(nn.Dense(self.hidden, kernel_init=_ORTHO(jnp.sqrt(2.0)))(others))
        pooled = jnp.mean(o, axis=-2)
        h = nn.tanh(nn.Dense(self.hidden, kernel_init=_ORTHO(jnp.sqrt(2.0)))(
            jnp.concatenate([s, pooled], axis=-1)))
        mean = nn.Dense(3, kernel_init=_ORTHO(0.01))(h)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (3,))
        value = nn.Dense(1, kernel_init=_ORTHO(1.0))(h).squeeze(-1)
        return mean, log_std, value


def log_prob(mean, log_std, action):
    """Tanh-squashed diagonal Gaussian log density. Summed over the action axis."""
    log_std = jnp.clip(log_std, -5.0, 2.0)
    std = jnp.exp(log_std)
    raw = jnp.arctanh(jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6))
    logp = -0.5 * ((raw - mean) ** 2 / (std ** 2) + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    logp = logp.sum(axis=-1)
    logp = logp - jnp.log(1.0 - action ** 2 + 1e-6).sum(axis=-1)
    return logp


def sample_action(key, mean, log_std):
    log_std = jnp.clip(log_std, -5.0, 2.0)
    raw = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    action = jnp.tanh(raw)
    return action, log_prob(mean, log_std, action)


def deterministic_action(mean):
    return jnp.tanh(mean)
