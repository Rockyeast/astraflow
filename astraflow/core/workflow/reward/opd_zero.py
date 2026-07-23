from astraflow.core.workflow.registry import register_reward


@register_reward("opd_zero")
def opd_zero_reward_fn(*args, **kwargs) -> float:
    """Return no environment reward; OPD supervision comes from the teacher."""
    return 0.0
