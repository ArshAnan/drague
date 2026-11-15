"""MAPPO: shared actor-critic per role, GAE, clipped surrogate."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from drague.algos.policy import ActorCritic, deterministic_action, log_prob, sample_action

GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
# The bonus is the pre-tanh Gaussian entropy, which grows without bound in log_std while the
# squashed actions saturate at +-1. With 0.01 it drove std up and pushed drones out of the arena
# (EXPERIMENTS.md, 2026-11-12), so it is off; the clipped surrogate sets the std on its own.
ENTROPY = 0.0
EPOCHS = 4
MINIBATCHES = 8
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 0.5
ALIVE_FEATURE = 9  # index of `alive` in the 10-value `self` observation (docs/game_spec.md, section 3)


def make_optimizer(learning_rate=LEARNING_RATE):
    """`learning_rate` may be a float or an optax schedule (counted in gradient steps)."""
    return optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(learning_rate))


def init_role(key, net: ActorCritic, self_feat, others):
    params = net.init(key, self_feat, others)
    opt = make_optimizer()
    return params, opt.init(params)


def _role_forward(net, params, self_feat, others):
    return net.apply(params, self_feat, others)


def gae(rewards, values, dones, bootstrap, gamma=GAMMA, lam=LAMBDA):
    """rewards, values, dones are (T, N). bootstrap is (N,)."""
    next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=0)
    next_values = next_values * (1.0 - dones)

    def step(carry, inputs):
        reward, value, next_value, done = inputs
        delta = reward + gamma * next_value - value
        carry = delta + gamma * lam * (1.0 - done) * carry
        return carry, carry

    _, adv = jax.lax.scan(
        step,
        jnp.zeros(rewards.shape[1:]),
        (rewards[::-1], values[::-1], next_values[::-1], dones[::-1]),
    )
    return adv[::-1]


def collect_rollout(key, state, pursuer_params, evader_params, horizon, env, net):
    """Scan `horizon` steps. Params are arguments so this can be jitted once per env."""
    P = env.cfg.n_pursuers

    def body(carry, _):
        key, state = carry
        key, k_step, k_p, k_e = jax.random.split(key, 4)
        obs = env.observe(state)
        p_mean, p_log_std, p_value = _role_forward(
            net, pursuer_params, obs["self"][:, :P], obs["others"][:, :P])
        e_mean, e_log_std, e_value = _role_forward(
            net, evader_params, obs["self"][:, P:P + 1], obs["others"][:, P:P + 1])
        p_act, p_logp = sample_action(k_p, p_mean, p_log_std)
        e_act, e_logp = sample_action(k_e, e_mean, e_log_std)
        actions = jnp.concatenate([p_act, e_act], axis=1)
        new_state, _, rewards, dones, info = env.step(k_step, state, actions)
        frame = {
            "self": obs["self"],
            "others": obs["others"],
            "actions": actions,
            "p_logp": p_logp,
            "e_logp": e_logp,
            "p_value": p_value,
            "e_value": e_value,
            "rewards": rewards,
            "dones": dones,
            **info,
        }
        return (key, new_state), frame

    (key, state), frames = jax.lax.scan(body, (key, state), None, int(horizon))
    obs = env.observe(state)
    _, _, p_boot = _role_forward(net, pursuer_params, obs["self"][:, :P], obs["others"][:, :P])
    _, _, e_boot = _role_forward(
        net, evader_params, obs["self"][:, P:P + 1], obs["others"][:, P:P + 1])
    return key, state, frames, p_boot, e_boot


def _ppo_loss(params, net, self_feat, others, actions, old_logp, adv, returns):
    mean, log_std, value = _role_forward(net, params, self_feat, others)
    new_logp = log_prob(mean, log_std, actions)
    ratio = jnp.exp(new_logp - old_logp)
    clipped = jnp.clip(ratio, 1.0 - CLIP, 1.0 + CLIP)
    # Crashed drones are frozen and their actions are ignored, so their samples carry no policy
    # signal. Mask them out of the policy terms (MAPPO "active masks"); the value head still
    # learns the team return from them.
    active = self_feat[..., ALIVE_FEATURE]
    n_active = jnp.maximum(jnp.sum(active), 1.0)
    active_mean = lambda x: jnp.sum(x * active) / n_active
    policy_loss = -active_mean(jnp.minimum(ratio * adv, clipped * adv))
    value_loss = jnp.mean((value - returns) ** 2)
    entropy = jnp.mean(jnp.sum(jnp.clip(log_std, -5.0, 2.0) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
    loss = policy_loss + 0.5 * value_loss - ENTROPY * entropy
    log_ratio = new_logp - old_logp
    return loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": active_mean(ratio - 1.0 - log_ratio),
        "clip_frac": active_mean((jnp.abs(ratio - 1.0) > CLIP).astype(jnp.float32)),
        "active_frac": jnp.mean(active),
        "mean_abs": active_mean(jnp.mean(jnp.abs(mean), axis=-1)),
    }


def ppo_update(params, opt_state, optimizer, key, self_feat, others, actions, old_logp, adv, returns,
               net, epochs=EPOCHS, minibatches=MINIBATCHES):
    """`epochs` passes over the batch, each shuffled into `minibatches` gradient steps.

    Inputs have three leading axes (T, W, A) for time, world and agent; they are flattened
    into one sample axis. Returns metrics averaged over every gradient step.
    """
    compiled = _compiled_ppo.setdefault(
        (id(optimizer), id(net), int(epochs), int(minibatches)),
        _compile_ppo(optimizer, net, int(epochs), int(minibatches)))
    return compiled(params, opt_state, key, self_feat, others, actions, old_logp, adv, returns)


def _compile_ppo(optimizer, net, epochs, minibatches):
    def update(params, opt_state, key, self_feat, others, actions, old_logp, adv, returns):
        flat = lambda x: x.reshape((-1,) + x.shape[3:])
        data = tuple(flat(x) for x in (self_feat, others, actions, old_logp, adv, returns))
        n = data[0].shape[0]
        n_mb = max(1, min(minibatches, n))
        size = n // n_mb

        def grad_step(carry, batch):
            params, opt_state = carry
            (loss, metrics), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(params, net, *batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), {**metrics, "loss": loss}

        def epoch(carry, k):
            perm = jax.random.permutation(k, n)[: n_mb * size].reshape(n_mb, size)
            batches = tuple(x[perm] for x in data)
            return jax.lax.scan(grad_step, carry, batches)

        (params, opt_state), metrics = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(key, epochs))
        return params, opt_state, jax.tree_util.tree_map(jnp.mean, metrics)

    return jax.jit(update)


_compiled_ppo: dict = {}


def advantages(rewards, values, dones, bootstrap, gamma=GAMMA, lam=LAMBDA):
    """Normalize advantages. rewards/values/dones are (T, N)."""
    adv = gae(rewards, values, dones.astype(jnp.float32), bootstrap, gamma, lam)
    returns = adv + values
    std = jnp.std(adv)
    adv = (adv - jnp.mean(adv)) / jnp.where(std > 1e-6, std, 1.0)
    return adv, returns


def update_pursuer(params, opt_state, optimizer, frames, bootstrap, net, epochs=EPOCHS, key=None,
                   gamma=GAMMA):
    P = frames["p_value"].shape[-1]
    rewards = jnp.broadcast_to(frames["rewards"][:, :, 0:1], frames["p_value"].shape)
    dones = jnp.broadcast_to(frames["dones"][:, :, None], frames["p_value"].shape)
    flat = lambda x: x.reshape(x.shape[0], -1)
    adv, returns = advantages(
        flat(rewards), flat(frames["p_value"]), flat(dones), bootstrap.reshape(-1), gamma)
    adv = adv.reshape(frames["p_value"].shape)
    returns = returns.reshape(frames["p_value"].shape)
    key = jax.random.PRNGKey(0) if key is None else key
    return ppo_update(
        params, opt_state, optimizer, key,
        frames["self"][:, :, :P], frames["others"][:, :, :P],
        frames["actions"][:, :, :P], frames["p_logp"], adv, returns, net, epochs)


def update_evader(params, opt_state, optimizer, frames, bootstrap, net, epochs=EPOCHS, key=None,
                  gamma=GAMMA):
    P = frames["self"].shape[-2] - 1
    values = frames["e_value"][..., 0]
    adv, returns = advantages(
        frames["rewards"][:, :, -1], values, frames["dones"], bootstrap[..., 0], gamma)
    adv, returns = adv[..., None], returns[..., None]
    key = jax.random.PRNGKey(0) if key is None else key
    return ppo_update(
        params, opt_state, optimizer, key,
        frames["self"][:, :, P:P + 1], frames["others"][:, :, P:P + 1],
        frames["actions"][:, :, P:P + 1], frames["e_logp"], adv, returns, net, epochs)


def deterministic_role_actions(params, net, self_feat, others):
    mean, _, _ = _role_forward(net, params, self_feat, others)
    return deterministic_action(mean)


# Jit the pieces that run every iteration. The env and network are static.
collect_rollout = jax.jit(collect_rollout, static_argnames=("horizon", "env", "net"))
advantages = jax.jit(advantages)
deterministic_role_actions = jax.jit(deterministic_role_actions, static_argnames=("net",))
