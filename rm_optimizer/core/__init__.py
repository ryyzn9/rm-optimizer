"""Core module exports."""

from rm_optimizer.core.base import PreferencePair, BaseRewardModel, EvaluationMetrics
from rm_optimizer.core.reward_model import BradleyTerryRM
from rm_optimizer.core.data_loader import PreferenceDataset, create_dataloader
from rm_optimizer.core.datasets import (
    download_dataset,
    list_datasets,
    load_preference_data,
    create_demo_dataset,
    DATASET_REGISTRY
)

__all__ = [
    "PreferencePair",
    "BaseRewardModel",
    "EvaluationMetrics",
    "BradleyTerryRM",
    "PreferenceDataset",
    "create_dataloader",
    "download_dataset",
    "list_datasets",
    "load_preference_data",
    "create_demo_dataset",
    "DATASET_REGISTRY",
]
