"""Voyager package entrypoint."""

from voyager.envs.registration import register_envs

__version__ = "0.0.0"

register_envs()

__all__ = ["__version__", "register_envs"]
