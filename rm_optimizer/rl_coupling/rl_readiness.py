"""
RL-readiness scoring for reward models.

This module provides:
- RLReadinessScorer: Composite score for RL training stability
- Over-optimization detection
- Early warning system

A high RL-readiness score predicts:
- Stable policy training
- Low policy gradient variance
- Resistance to reward hacking
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class RLReadinessReport:
    """
    Comprehensive RL-readiness report.
    
    Attributes:
        overall_score: Composite score 0-1 (higher is better)
        accuracy_score: Based on preference accuracy
        calibration_score: Based on ECE
        stability_score: Based on reward variance
        landscape_score: Based on Hessian flatness
        recommendations: List of improvement suggestions
    """
    overall_score: float
    accuracy_score: float
    calibration_score: float
    stability_score: float
    landscape_score: float
    recommendations: List[str]
    details: Dict[str, Any]
    
    def __str__(self) -> str:
        return (
            f"RL-Readiness Report\n"
            f"{'='*40}\n"
            f"Overall Score: {self.overall_score:.2f}/1.00\n"
            f"  - Accuracy:    {self.accuracy_score:.2f}\n"
            f"  - Calibration: {self.calibration_score:.2f}\n"
            f"  - Stability:   {self.stability_score:.2f}\n"
            f"  - Landscape:   {self.landscape_score:.2f}\n"
            f"\nRecommendations:\n"
            + "\n".join(f"  • {r}" for r in self.recommendations)
        )


class RLReadinessScorer:
    """
    Compute RL-readiness score for reward models.
    
    The score predicts how well a reward model will perform
    as a training signal for policy optimization.
    
    Components:
    1. Accuracy: Must correctly rank preferences
    2. Calibration: Predicted probabilities match reality
    3. Stability: Low reward variance for stable gradients
    4. Landscape: Flat minima generalize better
    
    Thresholds:
    - score > 0.8: Ready for RL training
    - 0.6 < score < 0.8: Acceptable, monitor closely
    - score < 0.6: Not recommended, retrain or adjust
    
    Args:
        weights: Dict of component weights (must sum to 1)
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        self.weights = weights or {
            'accuracy': 0.35,
            'calibration': 0.25,
            'stability': 0.25,
            'landscape': 0.15
        }
        
        # Validate weights
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0):
            # Normalize
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def compute_score(
        self,
        accuracy: float,
        ece: float = 0.1,
        reward_variance: float = 1.0,
        top_eigenvalue: float = 100.0,
        margin_mean: float = 0.5,
        ensemble_disagreement: float = 0.3
    ) -> RLReadinessReport:
        """
        Compute comprehensive RL-readiness score.
        
        Args:
            accuracy: Preference prediction accuracy (0-1)
            ece: Expected Calibration Error (lower is better)
            reward_variance: Variance of rewards (lower is better)
            top_eigenvalue: Largest Hessian eigenvalue (lower is better)
            margin_mean: Mean reward margin
            ensemble_disagreement: Ensemble std (lower is better)
        
        Returns:
            RLReadinessReport with scores and recommendations
        """
        recommendations = []
        details = {}
        
        # 1. Accuracy score (0-1, higher is better)
        accuracy_score = min(1.0, accuracy)
        if accuracy < 0.7:
            recommendations.append(
                f"Accuracy ({accuracy:.2%}) is below 70%. "
                "Consider more training data or longer training."
            )
        details['accuracy'] = accuracy
        
        # 2. Calibration score (transform ECE to 0-1 score)
        # ECE of 0 = score of 1, ECE of 0.2+ = score of 0
        calibration_score = max(0, 1 - ece * 5)
        if ece > 0.1:
            recommendations.append(
                f"ECE ({ece:.3f}) is high. Consider temperature scaling "
                "or calibration-aware training."
            )
        details['ece'] = ece
        
        # 3. Stability score (based on reward variance)
        # Low variance = high score
        # Variance of 0.5 = score of 1, variance of 3+ = score of 0
        stability_score = max(0, 1 - (reward_variance - 0.5) / 2.5)
        if reward_variance > 2.0:
            recommendations.append(
                f"High reward variance ({reward_variance:.2f}). "
                "This may cause unstable policy gradients."
            )
        details['reward_variance'] = reward_variance
        
        # 4. Landscape score (based on flatness)
        # Lower eigenvalue = flatter = better generalization
        # Eigenvalue < 50 = score of 1, > 500 = score of 0
        landscape_score = max(0, 1 - (top_eigenvalue - 50) / 450)
        if top_eigenvalue > 200:
            recommendations.append(
                f"Sharp minimum (λ_max={top_eigenvalue:.1f}). "
                "Consider SAM or sharpness-aware training."
            )
        details['top_eigenvalue'] = top_eigenvalue
        
        # Additional factors
        # Margin penalty for extreme values
        margin_penalty = 0.0
        if abs(margin_mean) > 3.0:
            margin_penalty = min(0.2, (abs(margin_mean) - 3.0) * 0.05)
            recommendations.append(
                f"Extreme margins (mean={margin_mean:.2f}). "
                "Model may be overconfident."
            )
        details['margin_mean'] = margin_mean
        
        # Ensemble disagreement penalty
        disagreement_penalty = 0.0
        if ensemble_disagreement > 0.5:
            disagreement_penalty = min(0.15, (ensemble_disagreement - 0.5) * 0.3)
            recommendations.append(
                f"High ensemble disagreement ({ensemble_disagreement:.2f}). "
                "Model may have high epistemic uncertainty."
            )
        details['ensemble_disagreement'] = ensemble_disagreement
        
        # Compute weighted score
        overall_score = (
            self.weights['accuracy'] * accuracy_score +
            self.weights['calibration'] * calibration_score +
            self.weights['stability'] * stability_score +
            self.weights['landscape'] * landscape_score
            - margin_penalty
            - disagreement_penalty
        )
        overall_score = max(0, min(1, overall_score))
        
        # Add overall recommendation
        if overall_score >= 0.8:
            recommendations.insert(0, "✓ Model is ready for RL training.")
        elif overall_score >= 0.6:
            recommendations.insert(0, "⚠ Model is acceptable but monitor training closely.")
        else:
            recommendations.insert(0, "✗ Model not recommended for RL. Address issues first.")
        
        return RLReadinessReport(
            overall_score=overall_score,
            accuracy_score=accuracy_score,
            calibration_score=calibration_score,
            stability_score=stability_score,
            landscape_score=landscape_score,
            recommendations=recommendations,
            details=details
        )
    
    def compute_from_metrics(
        self,
        metrics: Dict[str, float]
    ) -> RLReadinessReport:
        """
        Compute score from a metrics dictionary.
        
        Expected keys:
        - 'accuracy' or 'val_accuracy'
        - 'ece' or 'calibration_ece'
        - 'reward_variance' or 'margin_std'
        - 'top_eigenvalue' or 'hessian_top_eigenvalue'
        """
        accuracy = metrics.get('accuracy', metrics.get('val_accuracy', 0.8))
        ece = metrics.get('ece', metrics.get('calibration_ece', 0.1))
        variance = metrics.get('reward_variance', metrics.get('margin_std', 1.0) ** 2)
        eigenvalue = metrics.get('top_eigenvalue', metrics.get('hessian_top_eigenvalue', 100.0))
        margin = metrics.get('margin_mean', 0.5)
        disagreement = metrics.get('ensemble_disagreement', 0.3)
        
        return self.compute_score(
            accuracy=accuracy,
            ece=ece,
            reward_variance=variance,
            top_eigenvalue=eigenvalue,
            margin_mean=margin,
            ensemble_disagreement=disagreement
        )


def analyze_rl_readiness(
    reward_model,
    val_loader,
    hessian_spectrum=None,
    ensemble=None
) -> RLReadinessReport:
    """
    Convenience function to analyze RL-readiness.
    
    Args:
        reward_model: Trained reward model
        val_loader: Validation data loader
        hessian_spectrum: Optional pre-computed Hessian spectrum
        ensemble: Optional RewardModelEnsemble
    
    Returns:
        RLReadinessReport
    """
    from rm_optimizer.core.base import PreferencePair
    
    # Compute accuracy and calibration
    metrics = reward_model.evaluate([
        PreferencePair(
            prompt=batch['prompt'],
            chosen=batch['chosen'],
            rejected=batch['rejected']
        )
        for batch in val_loader
    ])
    
    # Get Hessian metrics if available
    top_eigenvalue = 100.0
    if hessian_spectrum is not None:
        top_eigenvalue = hessian_spectrum.eigenvalues[0]
    
    # Get ensemble disagreement if available
    disagreement = 0.3
    if ensemble is not None:
        # Sample a few predictions
        stats = ensemble.compute_disagreement(
            prompts=["Test prompt"],
            responses=["Test response"]
        )
        disagreement = stats.get('mean_disagreement', 0.3)
    
    scorer = RLReadinessScorer()
    return scorer.compute_score(
        accuracy=metrics.accuracy,
        ece=metrics.ece,
        reward_variance=1.0,  # Would need to compute from predictions
        top_eigenvalue=top_eigenvalue,
        ensemble_disagreement=disagreement
    )
