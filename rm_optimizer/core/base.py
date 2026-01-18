"""
Base classes and interfaces for the RM-Optimizer framework.

This module provides:
- PreferencePair: Data structure for preference comparisons
- BaseRewardModel: Abstract interface for reward models
- EvaluationMetrics: Standard evaluation metrics

Design Pattern: Abstract Base Class
Rationale: Enforces consistent API across different RM implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


@dataclass
class PreferencePair:
    """
    Data structure for a single preference comparison.
    
    Attributes:
        prompt: The input prompt/query
        chosen: The preferred (winning) response
        rejected: The dispreferred (losing) response
        margin: Optional ground truth margin score
        metadata: Optional additional information (source, timestamp, etc.)
    
    Design Decision: Use dataclass instead of dict
    Rationale: Type safety, IDE autocomplete, validation
    """
    prompt: str
    chosen: str
    rejected: str
    margin: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate inputs after initialization."""
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not self.chosen or not isinstance(self.chosen, str):
            raise ValueError("Chosen response must be a non-empty string")
        if not self.rejected or not isinstance(self.rejected, str):
            raise ValueError("Rejected response must be a non-empty string")
        if self.margin is not None and not isinstance(self.margin, (int, float)):
            raise ValueError("Margin must be a number if provided")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "margin": self.margin,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        """Create from dictionary."""
        return cls(
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            margin=data.get("margin"),
            metadata=data.get("metadata", {})
        )


@dataclass
class EvaluationMetrics:
    """
    Container for reward model evaluation metrics.
    
    Attributes:
        accuracy: Fraction of correct preference predictions
        ece: Expected Calibration Error
        brier: Brier score for probabilistic predictions
        auc: Area under ROC curve
        margin_correlation: Correlation with ground truth margins
    """
    accuracy: float
    ece: float = 0.0
    brier: float = 0.0
    auc: float = 0.0
    margin_correlation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "ece": self.ece,
            "brier": self.brier,
            "auc": self.auc,
            "margin_correlation": self.margin_correlation
        }
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"Accuracy: {self.accuracy:.4f} | "
            f"ECE: {self.ece:.4f} | "
            f"Brier: {self.brier:.4f} | "
            f"AUC: {self.auc:.4f}"
        )


class BaseRewardModel(ABC, nn.Module):
    """
    Abstract interface for reward models.
    
    All RM implementations must inherit from this and implement
    the abstract methods. This ensures consistent API across:
    - Different base architectures (DeBERTa, Llama, etc.)
    - Different training methods (Bradley-Terry, Plackett-Luce)
    - Different inference optimizations (vLLM, TensorRT)
    
    H100 Optimization Notes:
    - Use BF16 precision for Tensor Core acceleration
    - Enable FlashAttention-2 for memory efficiency
    - torch.compile() for 20-30% speedup
    """
    
    def __init__(
        self, 
        model_name: str, 
        device: str = "cuda",
        compile_model: bool = True,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.compile_model = compile_model
        self.use_flash_attention = use_flash_attention
        self._is_compiled = False
    
    @abstractmethod
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning scalar reward.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
        
        Returns:
            rewards: Scalar rewards, shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute preference loss for a batch.
        
        Args:
            chosen_ids: Token IDs for chosen responses
            chosen_mask: Attention mask for chosen responses
            rejected_ids: Token IDs for rejected responses
            rejected_mask: Attention mask for rejected responses
        
        Returns:
            loss: Scalar loss value
        """
        pass
    
    @abstractmethod
    def tokenize(
        self, 
        prompt: str, 
        response: str
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompt + response pair.
        
        Args:
            prompt: Input prompt
            response: Response text
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        pass
    
    def score_pair(
        self, 
        prompt: str, 
        response_a: str, 
        response_b: str
    ) -> float:
        """
        High-level API: compute reward margin between two responses.
        
        Args:
            prompt: Input prompt
            response_a: First response
            response_b: Second response
        
        Returns:
            margin: r(A) - r(B), positive if A is preferred
        """
        tokens_a = self.tokenize(prompt, response_a)
        tokens_b = self.tokenize(prompt, response_b)
        
        with torch.no_grad():
            reward_a = self.forward(
                tokens_a['input_ids'].to(self.device), 
                tokens_a['attention_mask'].to(self.device)
            )
            reward_b = self.forward(
                tokens_b['input_ids'].to(self.device), 
                tokens_b['attention_mask'].to(self.device)
            )
        
        return (reward_a - reward_b).item()
    
    def predict_preference(self, pair: PreferencePair) -> Tuple[bool, float]:
        """
        Predict which response is preferred.
        
        Args:
            pair: PreferencePair with prompt, chosen, rejected
        
        Returns:
            Tuple of (correct_prediction, margin)
        """
        margin = self.score_pair(pair.prompt, pair.chosen, pair.rejected)
        is_correct = margin > 0
        return is_correct, margin
    
    def evaluate(
        self, 
        pairs: List[PreferencePair],
        batch_size: int = 32
    ) -> EvaluationMetrics:
        """
        Evaluate reward model on a list of preference pairs.
        
        Args:
            pairs: List of PreferencePair objects
            batch_size: Batch size for evaluation
        
        Returns:
            EvaluationMetrics with accuracy and calibration
        """
        correct = 0
        total = 0
        margins = []
        probabilities = []
        
        self.eval()
        with torch.no_grad():
            for pair in pairs:
                is_correct, margin = self.predict_preference(pair)
                correct += int(is_correct)
                total += 1
                margins.append(margin)
                # Convert margin to probability via sigmoid
                prob = torch.sigmoid(torch.tensor(margin)).item()
                probabilities.append(prob)
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Compute ECE (Expected Calibration Error)
        ece = self._compute_ece(probabilities, [m > 0 for m in margins])
        
        # Compute Brier score
        brier = self._compute_brier(probabilities, [m > 0 for m in margins])
        
        return EvaluationMetrics(
            accuracy=accuracy,
            ece=ece,
            brier=brier
        )
    
    def _compute_ece(
        self, 
        probabilities: List[float], 
        labels: List[bool],
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        probs = np.array(probabilities)
        labs = np.array(labels, dtype=float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(probs[in_bin])
                avg_accuracy = np.mean(labs[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * np.sum(in_bin)
        
        return ece / len(probs) if len(probs) > 0 else 0.0
    
    def _compute_brier(
        self, 
        probabilities: List[float], 
        labels: List[bool]
    ) -> float:
        """Compute Brier score."""
        probs = np.array(probabilities)
        labs = np.array(labels, dtype=float)
        return np.mean((probs - labs) ** 2)
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def compile(self) -> None:
        """Apply torch.compile() for H100 optimization."""
        if not self._is_compiled and self.compile_model:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self._is_compiled = True
