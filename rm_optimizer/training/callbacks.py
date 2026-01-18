"""
Custom PyTorch Lightning callbacks for reward model analysis.

This module provides:
- HessianCallback: Periodic Hessian eigenspectrum computation
- CalibrationCallback: Track calibration metrics during training
- RLReadinessCallback: Monitor RL-readiness indicators

These callbacks enable real-time analysis of model properties
during training, not just at the end.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from typing import Optional, List, Dict, Any
import numpy as np


class HessianCallback(Callback):
    """
    Callback to compute Hessian eigenspectrum periodically during training.
    
    This enables tracking how the loss landscape evolves during training,
    not just the final landscape geometry.
    
    Args:
        compute_every_n_epochs: Compute Hessian every N epochs
        top_k_eigenvalues: Number of top eigenvalues to compute
        trace_samples: Number of samples for trace estimation
        log_eigenvalues: Whether to log individual eigenvalues
    
    Example:
        >>> callback = HessianCallback(compute_every_n_epochs=1)
        >>> trainer = pl.Trainer(callbacks=[callback])
    """
    
    def __init__(
        self,
        compute_every_n_epochs: int = 1,
        top_k_eigenvalues: int = 20,
        trace_samples: int = 50,
        log_eigenvalues: bool = True
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.top_k = top_k_eigenvalues
        self.trace_samples = trace_samples
        self.log_eigenvalues = log_eigenvalues
        
        # Store results
        self.eigenvalue_history: List[Dict[str, Any]] = []
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Compute Hessian at end of epoch if scheduled."""
        current_epoch = trainer.current_epoch
        
        if (current_epoch + 1) % self.compute_every_n_epochs != 0:
            return
        
        # Get validation dataloader for Hessian computation
        val_loader = trainer.val_dataloaders
        if val_loader is None:
            return
        
        try:
            from rm_optimizer.landscape.hessian import HessianAnalyzer
            
            analyzer = HessianAnalyzer(
                model=pl_module.model,
                device=pl_module.device
            )
            
            # Compute spectrum
            spectrum = analyzer.compute_spectrum(
                dataloader=val_loader,
                top_k=self.top_k,
                trace_samples=self.trace_samples
            )
            
            # Log metrics
            pl_module.log('hessian/top_eigenvalue', spectrum.eigenvalues[0])
            pl_module.log('hessian/trace', spectrum.trace_estimate)
            pl_module.log('hessian/condition_number', spectrum.condition_number)
            pl_module.log('hessian/effective_rank', spectrum.effective_rank)
            pl_module.log('hessian/sharpness_score', spectrum.sharpness_score)
            pl_module.log('hessian/flatness_index', spectrum.flatness_index)
            
            # Log individual eigenvalues if requested
            if self.log_eigenvalues:
                for i, eigenval in enumerate(spectrum.eigenvalues[:10]):
                    pl_module.log(f'hessian/eigenvalue_{i}', eigenval)
            
            # Store history
            self.eigenvalue_history.append({
                'epoch': current_epoch,
                'eigenvalues': spectrum.eigenvalues.tolist(),
                'trace': spectrum.trace_estimate,
                'condition_number': spectrum.condition_number
            })
            
            print(f"\n[Hessian] Epoch {current_epoch}: "
                  f"λ_max={spectrum.eigenvalues[0]:.4f}, "
                  f"trace={spectrum.trace_estimate:.4f}, "
                  f"κ={spectrum.condition_number:.4f}")
            
        except Exception as e:
            print(f"Warning: Hessian computation failed: {e}")


class CalibrationCallback(Callback):
    """
    Callback to track calibration metrics during training.
    
    Calibration is crucial for RL:
    - Well-calibrated RMs predict accurate preference probabilities
    - Poorly calibrated RMs cause policy gradient variance
    
    Args:
        compute_every_n_steps: Compute calibration every N steps
        n_bins: Number of bins for ECE computation
    """
    
    def __init__(
        self,
        compute_every_n_steps: int = 500,
        n_bins: int = 10
    ):
        super().__init__()
        self.compute_every_n_steps = compute_every_n_steps
        self.n_bins = n_bins
        
        # Accumulate predictions for calibration
        self.predictions: List[float] = []
        self.labels: List[int] = []
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Accumulate predictions for calibration computation."""
        if 'margin_mean' in outputs:
            margin = outputs['margin_mean'].item()
            prob = torch.sigmoid(torch.tensor(margin)).item()
            self.predictions.append(prob)
            self.labels.append(1)  # Chosen is always preferred in our data
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Compute and log calibration metrics."""
        if len(self.predictions) < 10:
            return
        
        probs = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Expected Calibration Error
        ece = self._compute_ece(probs, labels)
        
        # Brier Score
        brier = np.mean((probs - labels) ** 2)
        
        # Max Calibration Error
        mce = self._compute_mce(probs, labels)
        
        pl_module.log('calibration/ece', ece)
        pl_module.log('calibration/brier', brier)
        pl_module.log('calibration/mce', mce)
        
        # Reset for next epoch
        self.predictions.clear()
        self.labels.clear()
    
    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(probs[in_bin])
                avg_accuracy = np.mean(labels[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * np.sum(in_bin)
        
        return ece / len(probs)
    
    def _compute_mce(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        max_error = 0.0
        
        for i in range(self.n_bins):
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(probs[in_bin])
                avg_accuracy = np.mean(labels[in_bin])
                max_error = max(max_error, np.abs(avg_confidence - avg_accuracy))
        
        return max_error


class RLReadinessCallback(Callback):
    """
    Callback to monitor RL-readiness indicators during training.
    
    Tracks metrics that predict downstream RL training stability:
    - Reward variance (high variance = high policy gradient variance)
    - Margin distribution (very high/low margins indicate confidence issues)
    - Prediction consistency
    
    Args:
        compute_every_n_epochs: Compute RL metrics every N epochs
        variance_threshold: Threshold for variance warning
    """
    
    def __init__(
        self,
        compute_every_n_epochs: int = 1,
        variance_threshold: float = 2.0
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.variance_threshold = variance_threshold
        
        # Store metrics
        self.reward_history: List[Dict[str, float]] = []
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Compute RL-readiness score at end of validation."""
        current_epoch = trainer.current_epoch
        
        if (current_epoch + 1) % self.compute_every_n_epochs != 0:
            return
        
        # Get logged metrics
        metrics = trainer.callback_metrics
        
        # Extract relevant metrics
        margin_mean = metrics.get('val/margin_mean', 0.0)
        accuracy = metrics.get('val/accuracy', 0.0)
        
        # Compute RL-readiness score (0-1, higher is better)
        # Based on: accuracy, margin distribution, calibration
        
        accuracy_score = float(accuracy) if isinstance(accuracy, (int, float)) else accuracy.item()
        
        # Penalize extreme margins (indicates overconfidence)
        margin_penalty = 0.0
        if isinstance(margin_mean, torch.Tensor):
            margin_mean = margin_mean.item()
        if abs(margin_mean) > 3.0:
            margin_penalty = min(0.2, (abs(margin_mean) - 3.0) * 0.1)
        
        # Get calibration if available
        ece = metrics.get('calibration/ece', 0.1)
        if isinstance(ece, torch.Tensor):
            ece = ece.item()
        calibration_score = max(0, 1 - ece * 5)  # Penalize high ECE
        
        # Composite score
        rl_readiness = 0.5 * accuracy_score + 0.3 * calibration_score - margin_penalty
        rl_readiness = max(0, min(1, rl_readiness))
        
        pl_module.log('rl/readiness_score', rl_readiness)
        pl_module.log('rl/margin_mean', margin_mean)
        
        # Warning if variance too high
        if abs(margin_mean) > self.variance_threshold:
            print(f"\n⚠️  [RL Warning] High margin variance detected. "
                  f"margin_mean={margin_mean:.3f}. "
                  "This may cause RL training instability.")
        
        # Store history
        self.reward_history.append({
            'epoch': current_epoch,
            'rl_readiness': rl_readiness,
            'accuracy': accuracy_score,
            'margin_mean': margin_mean
        })


class GPUMemoryCallback(Callback):
    """
    Callback to monitor GPU memory usage.
    
    Helps ensure H100 80GB is being utilized efficiently.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """Log GPU memory usage."""
        if batch_idx % self.log_every_n_steps != 0:
            return
        
        if torch.cuda.is_available():
            # Memory allocated
            allocated = torch.cuda.memory_allocated() / 1e9
            # Memory reserved
            reserved = torch.cuda.memory_reserved() / 1e9
            # Max memory
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            
            pl_module.log('gpu/memory_allocated_gb', allocated)
            pl_module.log('gpu/memory_reserved_gb', reserved)
            pl_module.log('gpu/max_memory_gb', max_memory)
            
            # Utilization (assuming 80GB H100)
            utilization = allocated / 80.0 * 100
            pl_module.log('gpu/utilization_percent', utilization)
