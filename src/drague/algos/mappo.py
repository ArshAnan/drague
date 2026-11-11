"""MAPPO: shared actor-critic per role, GAE, clipped surrogate."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from drague.algos.policy import ActorCritic, deterministic_action, log_prob, sample_action

GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
ENTROPY = 0.01
EPOCHS = 2
LEARNING_RATE = 3e-4


def make_optimizer():
    return optax.adam(LEARNING_RATE)


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
    policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped * adv))
    value_loss = jnp.mean((value - returns) ** 2)
    entropy = jnp.mean(jnp.sum(jnp.clip(log_std, -5.0, 2.0) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
    loss = policy_loss + 0.5 * value_loss - ENTROPY * entropy
    return loss, {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


def ppo_epoch(params, opt_state, optimizer, self_feat, others, actions, old_logp, adv, returns, net):
    """One gradient step. The optimizer is closed over so its Python callables stay static."""
    compiled = _compiled_ppo.setdefault((id(optimizer), id(net)), _compile_ppo(optimizer, net))
    return compiled(params, opt_state, self_feat, others, actions, old_logp, adv, returns)


def _compile_ppo(optimizer, net):
    def epoch(params, opt_state, self_feat, others, actions, old_logp, adv, returns):
        (loss, metrics), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(
            params, net, self_feat, others, actions, old_logp, adv, returns)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {key: value for key, value in metrics.items()}
        metrics["loss"] = loss
        return params, opt_state, metrics

    return jax.jit(epoch)


_compiled_ppo: dict = {}


def advantages(rewards, values, dones, bootstrap):
    """Normalize advantages. rewards/values/dones are (T, N)."""
    adv = gae(rewards, values, dones.astype(jnp.float32), bootstrap)
    returns = adv + values
    std = jnp.std(adv)
    adv = (adv - jnp.mean(adv)) / jnp.where(std > 1e-6, std, 1.0)
    return adv, returns


def update_pursuer(params, opt_state, optimizer, frames, bootstrap, net, epochs=EPOCHS):
    P = frames["p_value"].shape[-1]
    rewards = jnp.broadcast_to(frames["rewards"][:, :, 0:1], frames["p_value"].shape)
    dones = jnp.broadcast_to(frames["dones"][:, :, None], frames["p_value"].shape)
    flat = lambda x: x.reshape(x.shape[0], -1)
    adv, returns = advantages(flat(rewards), flat(frames["p_value"]), flat(dones), bootstrap.reshape(-1))
    adv = adv.reshape(frames["p_value"].shape)
    returns = returns.reshape(frames["p_value"].shape)
    metrics = None
    for _ in range(epochs):
        params, opt_state, metrics = ppo_epoch(
            params, opt_state, optimizer,
            frames["self"][:, :, :P], frames["others"][:, :, :P],
            frames["actions"][:, :, :P], frames["p_logp"], adv, returns, net)
    return params, opt_state, metrics


def update_evader(params, opt_state, optimizer, frames, bootstrap, net, epochs=EPOCHS):
    P = frames["self"].shape[-2] - 1
    values = frames["e_value"][..., 0]
    adv, returns = advantages(
        frames["rewards"][:, :, -1], values, frames["dones"], bootstrap[..., 0])
    adv, returns = adv[..., None], returns[..., None]
    metrics = None
    for _ in range(epochs):
        params, opt_state, metrics = ppo_epoch(
            params, opt_state, optimizer,
            frames["self"][:, :, P:P + 1], frames["others"][:, :, P:P + 1],
            frames["actions"][:, :, P:P + 1], frames["e_logp"], adv, returns, net)
    return params, opt_state, metrics


def deterministic_role_actions(params, net, self_feat, others):
    mean, _, _ = _role_forward(net, params, self_feat, others)
    return deterministic_action(mean)


# Jit the pieces that run every iteration. The env and network are static.
collect_rollout = jax.jit(collect_rollout, static_argnames=("horizon", "env", "net"))
advantages = jax.jit(advantages)
deterministic_role_actions = jax.jit(deterministic_role_actions, static_argnames=("net",))
