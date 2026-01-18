"""
RM-Optimizer: Reward Model Analysis Framework

A comprehensive framework for analyzing reward models through:
- Loss landscape geometry (Hessian analysis)
- RL coupling dynamics (policy simulation)
- Data efficiency metrics (active learning)
"""

__version__ = "0.1.0"
__author__ = "RM-Optimizer Team"

from rm_optimizer.core.base import PreferencePair, BaseRewardModel
from rm_optimizer.core.reward_model import BradleyTerryRM

__all__ = [
    "PreferencePair",
    "BaseRewardModel", 
    "BradleyTerryRM",
]
