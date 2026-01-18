"""
Reward model ensemble for uncertainty quantification.

This module provides:
- RewardModelEnsemble: Manage multiple reward models
- Disagreement-based OOD detection
- Uncertainty estimation

H100 Advantage:
- 3×7B models fit in 80GB (42GB total)
- No model swapping needed
- Fast parallel scoring
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with uncertainty."""
    mean_reward: float
    std_reward: float
    individual_rewards: List[float]
    is_high_uncertainty: bool
    
    @property
    def uncertainty(self) -> float:
        return self.std_reward


class RewardModelEnsemble:
    """
    Ensemble of reward models for uncertainty quantification.
    
    Uses ensemble disagreement as a proxy for epistemic uncertainty.
    High disagreement indicates OOD data where the model is unreliable.
    
    H100 Memory Budget:
    - 3 × 7B models @ BF16 = 3 × 14GB = 42GB
    - Fits comfortably in 80GB H100
    - Can keep all models in memory simultaneously
    
    Args:
        models: List of reward models
        uncertainty_threshold: Threshold for high uncertainty
        device: Device for computation
    
    Example:
        >>> ensemble = RewardModelEnsemble(models=[rm1, rm2, rm3])
        >>> pred = ensemble.predict("What is 2+2?", "The answer is 4.")
        >>> print(f"Reward: {pred.mean_reward:.3f} ± {pred.std_reward:.3f}")
    """
    
    def __init__(
        self,
        models: List[nn.Module] = None,
        uncertainty_threshold: float = 0.5,
        device: str = "cuda"
    ):
        self.models = models or []
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device
        
        # Move all models to device
        for model in self.models:
            model.to(device)
            model.eval()
    
    def add_model(self, model: nn.Module) -> None:
        """Add a model to the ensemble."""
        model.to(self.device)
        model.eval()
        self.models.append(model)
    
    @property
    def size(self) -> int:
        """Number of models in ensemble."""
        return len(self.models)
    
    def predict(
        self,
        prompt: str,
        response: str
    ) -> EnsemblePrediction:
        """
        Get ensemble prediction with uncertainty.
        
        Args:
            prompt: Input prompt
            response: Response to score
        
        Returns:
            EnsemblePrediction with mean, std, and uncertainty flag
        """
        rewards = []
        
        with torch.no_grad():
            for model in self.models:
                reward = model.score_pair(prompt, response, "")
                rewards.append(reward)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        is_high_uncertainty = std_reward > self.uncertainty_threshold
        
        return EnsemblePrediction(
            mean_reward=mean_reward,
            std_reward=std_reward,
            individual_rewards=rewards,
            is_high_uncertainty=is_high_uncertainty
        )
    
    def predict_batch(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> List[EnsemblePrediction]:
        """Batch prediction for efficiency."""
        predictions = []
        
        for prompt, response in zip(prompts, responses):
            pred = self.predict(prompt, response)
            predictions.append(pred)
        
        return predictions
    
    def compute_disagreement(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> Dict[str, float]:
        """
        Compute ensemble disagreement statistics.
        
        Returns:
            Dict with mean/max disagreement and OOD fraction
        """
        predictions = self.predict_batch(prompts, responses)
        
        disagreements = [p.std_reward for p in predictions]
        ood_count = sum(1 for p in predictions if p.is_high_uncertainty)
        
        return {
            'mean_disagreement': np.mean(disagreements),
            'max_disagreement': np.max(disagreements),
            'std_disagreement': np.std(disagreements),
            'ood_fraction': ood_count / len(predictions),
            'ood_count': ood_count
        }
    
    def detect_ood(
        self,
        prompt: str,
        response: str
    ) -> bool:
        """
        Detect if input is out-of-distribution.
        
        Based on ensemble disagreement exceeding threshold.
        """
        pred = self.predict(prompt, response)
        return pred.is_high_uncertainty
    
    def calibrate_threshold(
        self,
        val_prompts: List[str],
        val_responses: List[str],
        target_ood_rate: float = 0.1
    ) -> float:
        """
        Calibrate uncertainty threshold on validation data.
        
        Sets threshold such that target_ood_rate fraction of
        validation data is flagged as OOD.
        """
        predictions = self.predict_batch(val_prompts, val_responses)
        uncertainties = sorted([p.std_reward for p in predictions])
        
        # Find threshold at (1 - target_ood_rate) percentile
        idx = int(len(uncertainties) * (1 - target_ood_rate))
        new_threshold = uncertainties[idx]
        
        self.uncertainty_threshold = new_threshold
        print(f"Calibrated threshold: {new_threshold:.4f}")
        
        return new_threshold
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of ensemble."""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in self.models
        )
        
        # Estimate memory (BF16 = 2 bytes per param)
        model_memory_gb = total_params * 2 / 1e9
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        
        return {
            'total_parameters': total_params,
            'estimated_model_memory_gb': model_memory_gb,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'utilization_percent': allocated / 80 * 100  # H100 80GB
        }


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    model_class,
    device: str = "cuda",
    **model_kwargs
) -> RewardModelEnsemble:
    """
    Create ensemble from saved checkpoints.
    
    Args:
        checkpoint_paths: Paths to model checkpoints
        model_class: Class to instantiate (e.g., BradleyTerryRM)
        device: Device for models
        **model_kwargs: Additional model arguments
    
    Returns:
        RewardModelEnsemble
    """
    models = []
    
    for path in checkpoint_paths:
        print(f"Loading {path}...")
        model = model_class.from_pretrained(path, device=device, **model_kwargs)
        models.append(model)
    
    return RewardModelEnsemble(models=models, device=device)
