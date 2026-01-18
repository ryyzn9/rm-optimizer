"""
Reward margin analysis for data efficiency.

This module provides:
- MarginAnalyzer: Analyze reward margin distribution
- CurriculumSampler: Sample based on difficulty

Key insight:
- Easy pairs (high margin): Low gradient signal
- Hard pairs (low margin): Noisy, possibly mislabeled
- Medium pairs: Most informative for training
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MarginStatistics:
    """Statistics about reward margin distribution."""
    mean: float
    std: float
    min: float
    max: float
    
    # Percentiles
    p25: float
    p50: float  # median
    p75: float
    
    # Fraction in each category
    easy_fraction: float  # margin > 0.7
    medium_fraction: float  # 0.2 <= margin <= 0.7
    hard_fraction: float  # margin < 0.2
    
    def __str__(self) -> str:
        return (
            f"Margin Statistics:\n"
            f"  Mean: {self.mean:.3f} ± {self.std:.3f}\n"
            f"  Range: [{self.min:.3f}, {self.max:.3f}]\n"
            f"  Median: {self.p50:.3f}\n"
            f"  Distribution: Easy {self.easy_fraction:.1%}, "
            f"Medium {self.medium_fraction:.1%}, Hard {self.hard_fraction:.1%}"
        )


class MarginAnalyzer:
    """
    Analyze reward margin distribution.
    
    Margins tell us about data quality and model confidence:
    - High margins: Clear preferences, model confident
    - Low margins: Ambiguous or difficult comparisons
    - Negative margins: Model disagrees with label (possible errors)
    
    Args:
        model: Reward model for computing margins
        easy_threshold: Margin above which pairs are "easy"
        hard_threshold: Margin below which pairs are "hard"
    """
    
    def __init__(
        self,
        model=None,
        easy_threshold: float = 0.7,
        hard_threshold: float = 0.2
    ):
        self.model = model
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
    
    def compute_margins(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str]
    ) -> np.ndarray:
        """
        Compute reward margins for preference pairs.
        
        Margin = r(chosen) - r(rejected)
        Positive margin = correct prediction
        """
        margins = []
        
        for p, c, r in zip(prompts, chosen, rejected):
            margin = self.model.score_pair(p, c, r)
            margins.append(margin)
        
        return np.array(margins)
    
    def analyze(
        self,
        margins: np.ndarray
    ) -> MarginStatistics:
        """
        Compute statistics from margins.
        
        Args:
            margins: Array of reward margins
        
        Returns:
            MarginStatistics object
        """
        # Basic statistics
        mean = np.mean(margins)
        std = np.std(margins)
        min_val = np.min(margins)
        max_val = np.max(margins)
        
        # Percentiles
        p25, p50, p75 = np.percentile(margins, [25, 50, 75])
        
        # Category fractions
        easy_count = np.sum(margins > self.easy_threshold)
        hard_count = np.sum(margins < self.hard_threshold)
        medium_count = len(margins) - easy_count - hard_count
        
        return MarginStatistics(
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            p25=p25,
            p50=p50,
            p75=p75,
            easy_fraction=easy_count / len(margins),
            medium_fraction=medium_count / len(margins),
            hard_fraction=hard_count / len(margins)
        )
    
    def find_mislabeled(
        self,
        margins: np.ndarray,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        threshold: float = -0.5
    ) -> List[Dict]:
        """
        Find potentially mislabeled examples.
        
        Examples where model strongly disagrees with label
        (large negative margin) may be mislabeled.
        
        Args:
            margins: Precomputed margins
            prompts, chosen, rejected: Original data
            threshold: Margin below which to flag
        
        Returns:
            List of suspicious examples
        """
        suspicious = []
        
        for i, margin in enumerate(margins):
            if margin < threshold:
                suspicious.append({
                    'index': i,
                    'margin': margin,
                    'prompt': prompts[i],
                    'labeled_chosen': chosen[i],
                    'labeled_rejected': rejected[i]
                })
        
        # Sort by margin (most negative first)
        suspicious.sort(key=lambda x: x['margin'])
        
        return suspicious


class CurriculumSampler:
    """
    Curriculum learning sampler based on margin difficulty.
    
    Strategy:
    - Early training: Focus on easy examples (clear signal)
    - Mid training: Mix in medium examples (informative)
    - Late training: Add hard examples (fine-tuning)
    
    Alternative: Always oversample medium-margin examples.
    
    Args:
        margins: Precomputed margins for training data
        easy_weight: Sampling weight for easy examples
        medium_weight: Sampling weight for medium examples
        hard_weight: Sampling weight for hard examples
    """
    
    def __init__(
        self,
        margins: np.ndarray,
        easy_weight: float = 0.2,
        medium_weight: float = 0.6,
        hard_weight: float = 0.2,
        easy_threshold: float = 0.7,
        hard_threshold: float = 0.2
    ):
        self.margins = margins
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        
        # Categorize indices
        self.easy_indices = np.where(margins > easy_threshold)[0]
        self.hard_indices = np.where(margins < hard_threshold)[0]
        self.medium_indices = np.where(
            (margins >= hard_threshold) & (margins <= easy_threshold)
        )[0]
        
        # Set weights
        self.weights = {
            'easy': easy_weight,
            'medium': medium_weight,
            'hard': hard_weight
        }
        
        print(f"CurriculumSampler initialized:")
        print(f"  Easy: {len(self.easy_indices)} examples")
        print(f"  Medium: {len(self.medium_indices)} examples")
        print(f"  Hard: {len(self.hard_indices)} examples")
    
    def sample(
        self,
        n: int,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample indices according to curriculum.
        
        Args:
            n: Number of samples
            epoch: Current epoch (for curriculum schedule)
            total_epochs: Total epochs (for curriculum schedule)
        
        Returns:
            Array of sampled indices
        """
        # Adjust weights based on training progress
        if epoch is not None and total_epochs is not None:
            progress = epoch / total_epochs
            weights = self._get_curriculum_weights(progress)
        else:
            weights = self.weights
        
        # Sample from each category
        samples = []
        
        for category, weight in weights.items():
            indices = getattr(self, f'{category}_indices')
            if len(indices) == 0:
                continue
            
            n_samples = int(n * weight)
            if n_samples > 0:
                sampled = np.random.choice(indices, size=n_samples, replace=True)
                samples.extend(sampled)
        
        # Shuffle and return
        samples = np.array(samples)
        np.random.shuffle(samples)
        
        return samples[:n]
    
    def _get_curriculum_weights(self, progress: float) -> Dict[str, float]:
        """
        Get curriculum weights based on training progress.
        
        Early: More easy examples
        Late: More hard examples
        """
        if progress < 0.3:
            # Early: focus on easy
            return {'easy': 0.5, 'medium': 0.4, 'hard': 0.1}
        elif progress < 0.7:
            # Middle: focus on medium
            return {'easy': 0.2, 'medium': 0.6, 'hard': 0.2}
        else:
            # Late: include more hard
            return {'easy': 0.1, 'medium': 0.5, 'hard': 0.4}
    
    def get_sampling_weights(self) -> np.ndarray:
        """
        Get per-sample weights for weighted random sampling.
        
        Can be used with PyTorch WeightedRandomSampler.
        """
        weights = np.zeros(len(self.margins))
        
        # Assign weights based on category
        for idx in self.easy_indices:
            weights[idx] = self.weights['easy'] / len(self.easy_indices)
        for idx in self.medium_indices:
            weights[idx] = self.weights['medium'] / len(self.medium_indices)
        for idx in self.hard_indices:
            weights[idx] = self.weights['hard'] / len(self.hard_indices)
        
        # Normalize
        weights /= weights.sum()
        
        return weights
