"""
Framework for comparing optimizers across geometric metrics.

This module provides:
- OptimizerResult: Results for a single optimizer
- OptimizerComparison: Run controlled experiments

Protocol:
1. Same model architecture
2. Same dataset
3. Same hyperparameters (except optimizer)
4. Multiple seeds for statistical significance
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class OptimizerResult:
    """
    Results for a single optimizer training run.
    
    Stores:
    - Training metrics (accuracy, loss)
    - Landscape metrics (Hessian spectrum)
    - Calibration metrics (ECE, Brier)
    - RL coupling metrics (readiness score)
    """
    optimizer_name: str
    seed: int
    
    # Training metrics
    final_train_accuracy: float
    final_val_accuracy: float
    final_train_loss: float
    final_val_loss: float
    training_time_seconds: float
    
    # Landscape metrics
    top_eigenvalue: float = 0.0
    trace_estimate: float = 0.0
    condition_number: float = 0.0
    effective_rank: float = 0.0
    sharpness_score: float = 0.0
    flatness_index: float = 0.0
    
    # Calibration metrics
    calibration_ece: float = 0.0
    calibration_brier: float = 0.0
    
    # RL coupling metrics
    rl_readiness_score: float = 0.0
    policy_grad_variance: float = 0.0
    
    # Full eigenvalue spectrum (optional)
    eigenvalues: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerResult":
        """Create from dictionary."""
        return cls(**data)


class OptimizerComparison:
    """
    Run controlled optimizer comparison experiments.
    
    Ensures fair comparison by:
    - Using identical model architecture
    - Using identical dataset splits
    - Using identical hyperparameters (except optimizer)
    - Running multiple seeds for statistical significance
    
    Args:
        model_name: HuggingFace model identifier
        data_path: Path to preference data
        optimizers: List of optimizer names to compare
        num_seeds: Number of random seeds per optimizer
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Base learning rate
        output_dir: Directory for results
    
    Example:
        >>> comparison = OptimizerComparison(
        ...     model_name="microsoft/deberta-v3-large",
        ...     optimizers=["adam", "muon", "lion"],
        ...     num_seeds=3
        ... )
        >>> comparison.run_experiment()
        >>> comparison.generate_report()
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        data_path: Optional[str] = None,
        optimizers: List[str] = None,
        num_seeds: int = 3,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        output_dir: str = "outputs/optimizer_comparison",
        compute_hessian: bool = True,
        hessian_top_k: int = 20
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.optimizers = optimizers or ["adam", "adamw", "sgd", "muon", "lion"]
        self.num_seeds = num_seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.compute_hessian = compute_hessian
        self.hessian_top_k = hessian_top_k
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results: Dict[str, List[OptimizerResult]] = {
            opt: [] for opt in self.optimizers
        }
    
    def run_experiment(self) -> None:
        """
        Run full comparison experiment.
        
        H100 Timeline (7B model, 100K samples):
        - Training per optimizer: ~15 min
        - Hessian analysis: ~5 min
        - Total for 5 optimizers × 3 seeds: ~5 hours
        """
        print("=" * 60)
        print("OPTIMIZER COMPARISON EXPERIMENT")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Optimizers: {self.optimizers}")
        print(f"Seeds: {self.num_seeds}")
        print(f"Epochs: {self.epochs}")
        print("=" * 60)
        
        total_runs = len(self.optimizers) * self.num_seeds
        current_run = 0
        
        for optimizer_name in self.optimizers:
            print(f"\n{'='*60}")
            print(f"OPTIMIZER: {optimizer_name.upper()}")
            print(f"{'='*60}")
            
            for seed in range(self.num_seeds):
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] {optimizer_name} seed={seed}")
                
                try:
                    result = self._train_and_analyze(optimizer_name, seed)
                    self.results[optimizer_name].append(result)
                    
                    # Save intermediate results
                    self._save_intermediate_results()
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
        
        # Final save
        self._save_final_results()
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
    
    def _train_and_analyze(
        self,
        optimizer_name: str,
        seed: int
    ) -> OptimizerResult:
        """Train model and compute all metrics."""
        import torch
        import random
        import numpy as np
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        start_time = time.time()
        
        # Import here to avoid circular imports
        from rm_optimizer.core.reward_model import BradleyTerryRM
        from rm_optimizer.training.lightning_module import train_reward_model
        
        # Train model
        print(f"  Training {optimizer_name}...")
        checkpoint_path = train_reward_model(
            model_name=self.model_name,
            optimizer=optimizer_name,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            data_path=self.data_path,
            checkpoint_dir=str(self.output_dir / f"{optimizer_name}_seed{seed}"),
            use_wandb=False,  # Disable for comparison runs
        )
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.1f}s")
        
        # Load trained model for analysis
        model = BradleyTerryRM.from_pretrained(
            checkpoint_path.replace('.ckpt', ''),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Compute metrics (placeholders - actual values from training logs)
        result = OptimizerResult(
            optimizer_name=optimizer_name,
            seed=seed,
            final_train_accuracy=0.0,  # From training logs
            final_val_accuracy=0.0,
            final_train_loss=0.0,
            final_val_loss=0.0,
            training_time_seconds=training_time
        )
        
        # Compute Hessian if enabled
        if self.compute_hessian:
            print(f"  Computing Hessian spectrum...")
            try:
                from rm_optimizer.landscape.hessian import HessianAnalyzer
                from rm_optimizer.core.data_loader import create_dataloader
                
                # Get data loader for Hessian
                if self.data_path:
                    loader = create_dataloader(
                        self.data_path,
                        tokenizer=model.tokenizer,
                        batch_size=8,
                        shuffle=False
                    )
                    
                    analyzer = HessianAnalyzer(model)
                    spectrum = analyzer.compute_spectrum(
                        loader,
                        top_k=self.hessian_top_k
                    )
                    
                    result.top_eigenvalue = float(spectrum.eigenvalues[0])
                    result.trace_estimate = spectrum.trace_estimate
                    result.condition_number = spectrum.condition_number
                    result.effective_rank = spectrum.effective_rank
                    result.sharpness_score = spectrum.sharpness_score
                    result.flatness_index = spectrum.flatness_index
                    result.eigenvalues = spectrum.eigenvalues.tolist()
                    
                    print(f"  λ_max={result.top_eigenvalue:.4f}, "
                          f"trace={result.trace_estimate:.4f}")
            except Exception as e:
                print(f"  Hessian computation failed: {e}")
        
        return result
    
    def _save_intermediate_results(self) -> None:
        """Save results after each run."""
        results_dict = {}
        for opt, results in self.results.items():
            results_dict[opt] = [r.to_dict() for r in results]
        
        with open(self.output_dir / "intermediate_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def _save_final_results(self) -> None:
        """Save final results."""
        results_dict = {}
        for opt, results in self.results.items():
            results_dict[opt] = [r.to_dict() for r in results]
        
        with open(self.output_dir / "final_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate summary report.
        
        Returns DataFrame with mean and std for each metric.
        """
        rows = []
        
        for opt_name, results in self.results.items():
            if not results:
                continue
            
            # Aggregate metrics across seeds
            metrics = {
                'optimizer': opt_name,
                'val_accuracy_mean': np.mean([r.final_val_accuracy for r in results]),
                'val_accuracy_std': np.std([r.final_val_accuracy for r in results]),
                'top_eigenvalue_mean': np.mean([r.top_eigenvalue for r in results]),
                'top_eigenvalue_std': np.std([r.top_eigenvalue for r in results]),
                'condition_number_mean': np.mean([r.condition_number for r in results]),
                'sharpness_mean': np.mean([r.sharpness_score for r in results]),
                'flatness_mean': np.mean([r.flatness_index for r in results]),
                'training_time_mean': np.mean([r.training_time_seconds for r in results]),
            }
            rows.append(metrics)
        
        df = pd.DataFrame(rows)
        
        # Save report
        df.to_csv(self.output_dir / "summary_report.csv", index=False)
        print("\nSummary Report:")
        print(df.to_string(index=False))
        
        return df
    
    def load_results(self, path: Optional[str] = None) -> None:
        """Load results from file."""
        path = path or (self.output_dir / "final_results.json")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for opt, results in data.items():
            self.results[opt] = [OptimizerResult.from_dict(r) for r in results]
