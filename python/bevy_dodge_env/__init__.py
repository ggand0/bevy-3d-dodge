"""Gymnasium environment wrapper for Bevy 3D dodge game."""

from bevy_dodge_env.environment import BevyDodgeEnv

__version__ = "0.1.0"
__all__ = ["BevyDodgeEnv"]

# Lazy imports for optional dependencies
def __getattr__(name):
    if name in ("make_env", "make_vec_env"):
        from bevy_dodge_env.vec_env import make_env, make_vec_env
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
