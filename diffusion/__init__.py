# diffusion/__init__.py
from .schedule import DiffusionSchedule, make_beta_schedule

__all__ = ["DiffusionSchedule", "make_beta_schedule"]
