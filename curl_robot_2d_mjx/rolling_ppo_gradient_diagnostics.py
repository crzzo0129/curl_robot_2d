"""Observe existing PPO gradients without another backward pass or file patches."""
from types import SimpleNamespace


def train_with_gradient_diagnostics(ppo, *, enabled=False, **kwargs):
    if not enabled:
        return ppo.train(**kwargs)
    import jax.numpy as jp
    import optax
    max_grad_norm = kwargs.get('max_grad_norm')
    original = ppo.gradients
    factory = original.loss_and_pgrad

    def instrumented_factory(*args, **options):
        if options.get('has_aux') is not True:
            raise ValueError('PPO gradient diagnostics requires auxiliary loss metrics')
        evaluate = factory(*args, **options)
        def evaluate_with_metrics(*inputs, **kw):
            (loss, metrics), gradients = evaluate(*inputs, **kw)
            policy_norm = optax.global_norm(gradients.policy)
            value_norm = optax.global_norm(gradients.value)
            norm = optax.global_norm(gradients)
            scale = (jp.minimum(1., max_grad_norm / jp.maximum(norm, 1e-12))
                     if max_grad_norm is not None else jp.asarray(1.))
            return (loss, {**metrics,
                'actor_grad_norm':policy_norm, 'critic_grad_norm':value_norm,
                'global_grad_norm':norm, 'grad_clip_scale':scale,
                'grad_clip_fraction':(scale < 1.).astype(jp.float32)}), gradients
        return evaluate_with_metrics

    # Replace only this trainer's module reference for the duration of its call.
    # The shared Brax gradients module and its implementation stay untouched.
    ppo.gradients = SimpleNamespace(**{**vars(original), 'loss_and_pgrad':instrumented_factory})
    try:
        return ppo.train(**kwargs)
    finally:
        ppo.gradients = original
