"""Shared actor-critic, independent of team size.

Other drones are split by the `same_team` flag and encoded by two separate per-row networks,
then pooled within each group. One pool mixing teammates and opponents (the first version) made
"a teammate is close" and "the evader is close" look alike, so learning to avoid teammates also
taught pursuers to back off the evader (EXPERIMENTS.md, 2026-11-12).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

_ORTHO = nn.initializers.orthogonal
SAME_TEAM_FEATURE = 8  # index of `same_team` in each 10-value `others` row (docs/game_spec.md, section 3)


def _dense(features, scale=jnp.sqrt(2.0), name=None):
    return nn.Dense(features, kernel_init=_ORTHO(scale), name=name)


def _masked_pools(x, mask):
    """Mean and max over the row axis, using only rows where mask is 1. Empty groups give zeros."""
    m = mask[..., None]
    count = jnp.sum(m, axis=-2)
    mean = jnp.sum(x * m, axis=-2) / jnp.maximum(count, 1.0)
    big = jnp.max(jnp.where(m > 0, x, -jnp.inf), axis=-2)
    big = jnp.where(count > 0, big, 0.0)
    return jnp.concatenate([mean, big], axis=-1)


class ActorCritic(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, self_feat, others):
        # self_feat (..., 10), others (..., M, 10). Leading batch axes are free.
        h = self.hidden
        s = nn.tanh(_dense(h, name="self_enc")(self_feat))
        mate = others[..., SAME_TEAM_FEATURE]
        groups = []
        for name, mask in (("mate", mate), ("opp", 1.0 - mate)):
            x = nn.tanh(_dense(h, name=f"{name}_enc0")(others))
            x = nn.tanh(_dense(h, name=f"{name}_enc1")(x))
            groups.append(_masked_pools(x, mask))
        z = jnp.concatenate([s, *groups], axis=-1)
        z = nn.tanh(_dense(h, name="trunk0")(z))
        z = nn.tanh(_dense(h, name="trunk1")(z))
        mean = _dense(3, 0.01, name="mean_head")(z)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (3,))
        value = _dense(1, 1.0, name="value_head")(z).squeeze(-1)
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
