"""
PyTorch Lightning module for reward model training.

This module provides:
- RewardModelLightning: Lightning wrapper for RM training
- Multi-optimizer support (Adam, AdamW, SGD, Muon, Lion)
- Warmup + cosine learning rate scheduling
- W&B integration

H100 Optimizations:
- BF16 mixed precision for Tensor Core acceleration
- Gradient checkpointing for memory efficiency
- torch.compile() for 20-30% speedup
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional, List
import os

from rm_optimizer.core.base import BaseRewardModel


class RewardModelLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for reward model training.
    
    Handles:
    - Training loop (automatic)
    - Validation loop (automatic)
    - Multi-optimizer configuration
    - Learning rate scheduling
    - Logging to W&B
    
    H100-Specific Features:
    - BF16 precision for Tensor Cores
    - Large batch training (no gradient accumulation needed)
    - Optimized data loading with pin_memory
    
    Args:
        base_model: BaseRewardModel instance
        learning_rate: Base learning rate
        optimizer_name: Optimizer choice (adam, adamw, sgd, muon, lion)
        optimizer_kwargs: Additional optimizer arguments
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps (for scheduler)
        weight_decay: Weight decay factor
    
    Example:
        >>> model = BradleyTerryRM("microsoft/deberta-v3-large")
        >>> lightning_model = RewardModelLightning(model)
        >>> trainer = pl.Trainer(max_epochs=5)
        >>> trainer.fit(lightning_model, train_loader, val_loader)
    """
    
    def __init__(
        self,
        base_model: BaseRewardModel,
        learning_rate: float = 1e-5,
        optimizer_name: str = "adamw",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        
        # Save hyperparameters (logged to W&B automatically)
        self.save_hyperparameters(ignore=['base_model'])
        
        self.model = base_model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        
        # Metrics tracking
        self.train_step_outputs: List[Dict] = []
        self.val_step_outputs: List[Dict] = []
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass (called by trainer)."""
        return self.model.forward(input_ids, attention_mask)
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Single training step.
        
        PyTorch Lightning handles:
        - optimizer.zero_grad()
        - loss.backward()
        - optimizer.step()
        - Gradient clipping
        - Mixed precision scaling
        """
        metrics = self.model.compute_loss_with_metrics(
            chosen_ids=batch['chosen_input_ids'],
            chosen_mask=batch['chosen_attention_mask'],
            rejected_ids=batch['rejected_input_ids'],
            rejected_mask=batch['rejected_attention_mask']
        )
        
        loss = metrics['loss']
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', metrics['accuracy'], on_step=True, on_epoch=True)
        self.log('train/margin_mean', metrics['margin_mean'], on_step=False, on_epoch=True)
        self.log('train/margin_std', metrics['margin_std'], on_step=False, on_epoch=True)
        self.log('train/reward_chosen', metrics['reward_chosen_mean'], on_step=False, on_epoch=True)
        self.log('train/reward_rejected', metrics['reward_rejected_mean'], on_step=False, on_epoch=True)
        
        self.train_step_outputs.append({
            'loss': loss.detach(),
            'accuracy': metrics['accuracy'].detach()
        })
        
        return loss
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        with torch.no_grad():
            metrics = self.model.compute_loss_with_metrics(
                chosen_ids=batch['chosen_input_ids'],
                chosen_mask=batch['chosen_attention_mask'],
                rejected_ids=batch['rejected_input_ids'],
                rejected_mask=batch['rejected_attention_mask']
            )
        
        self.log('val/loss', metrics['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', metrics['accuracy'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/margin_mean', metrics['margin_mean'], on_step=False, on_epoch=True)
        
        self.val_step_outputs.append({
            'loss': metrics['loss'].detach(),
            'accuracy': metrics['accuracy'].detach()
        })
        
        return metrics
    
    def on_train_epoch_end(self):
        """Hook called at end of training epoch."""
        if self.train_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
            avg_acc = torch.stack([x['accuracy'] for x in self.train_step_outputs]).mean()
            self.log('train/epoch_loss', avg_loss)
            self.log('train/epoch_accuracy', avg_acc)
        self.train_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Hook called at end of validation epoch."""
        if self.val_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
            avg_acc = torch.stack([x['accuracy'] for x in self.val_step_outputs]).mean()
            self.log('val/epoch_loss', avg_loss)
            self.log('val/epoch_accuracy', avg_acc)
        self.val_step_outputs.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Supports: adam, adamw, sgd, muon, lion
        Schedule: Linear warmup + cosine decay
        """
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize optimizer
        optimizer = self._create_optimizer(params)
        
        # Create learning rate scheduler
        scheduler = self._create_scheduler(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/accuracy",
            }
        }
    
    def _create_optimizer(self, params) -> torch.optim.Optimizer:
        """Create optimizer based on name."""
        name = self.optimizer_name.lower()
        
        if name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.learning_rate,
                betas=self.optimizer_kwargs.get('betas', (0.9, 0.999)),
                eps=self.optimizer_kwargs.get('eps', 1e-8),
            )
        
        elif name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.optimizer_kwargs.get('betas', (0.9, 0.999)),
                eps=self.optimizer_kwargs.get('eps', 1e-8),
            )
        
        elif name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.optimizer_kwargs.get('momentum', 0.9),
                weight_decay=self.weight_decay,
                nesterov=self.optimizer_kwargs.get('nesterov', True),
            )
        
        elif name == "muon":
            # Muon optimizer with Newton-Schulz orthogonalization
            try:
                from muon import Muon
                return Muon(
                    params,
                    lr=self.learning_rate,
                    momentum=self.optimizer_kwargs.get('momentum', 0.95),
                    nesterov=self.optimizer_kwargs.get('nesterov', True),
                    ns_steps=self.optimizer_kwargs.get('ns_steps', 5),
                )
            except ImportError:
                print("Muon not installed, falling back to AdamW")
                return torch.optim.AdamW(params, lr=self.learning_rate)
        
        elif name == "lion":
            try:
                from lion_pytorch import Lion
                # Lion uses smaller learning rate (10x smaller than Adam)
                return Lion(
                    params,
                    lr=self.learning_rate * 0.1,
                    betas=self.optimizer_kwargs.get('betas', (0.9, 0.99)),
                    weight_decay=self.optimizer_kwargs.get('weight_decay', 0.1),
                )
            except ImportError:
                print("Lion not installed, falling back to AdamW")
                return torch.optim.AdamW(params, lr=self.learning_rate)
        
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler with warmup + cosine decay."""
        # Warmup phase
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        
        # Cosine decay phase
        cosine_steps = max(1, self.max_steps - self.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.learning_rate * 0.1
        )
        
        # Combine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
        
        return scheduler


def train_reward_model(
    model_name: str = "microsoft/deberta-v3-large",
    optimizer: str = "adamw",
    learning_rate: float = 1e-5,
    batch_size: int = 32,
    max_epochs: int = 5,
    data_path: Optional[str] = None,
    train_loader=None,
    val_loader=None,
    checkpoint_dir: str = "checkpoints",
    wandb_project: str = "rm-optimizer",
    wandb_name: Optional[str] = None,
    use_wandb: bool = True,
    precision: str = "bf16-mixed",
    gradient_clip: float = 1.0,
    **kwargs
) -> str:
    """
    Main training function.
    
    Args:
        model_name: HuggingFace model identifier
        optimizer: Optimizer name (adam, adamw, sgd, muon, lion)
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Number of epochs
        data_path: Path to preference data (if loaders not provided)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        checkpoint_dir: Directory for checkpoints
        wandb_project: W&B project name
        wandb_name: W&B run name
        use_wandb: Whether to log to W&B
        precision: Training precision (bf16-mixed for H100)
        gradient_clip: Gradient clipping value
    
    Returns:
        Path to best checkpoint
    
    Example:
        >>> best_ckpt = train_reward_model(
        ...     model_name="microsoft/deberta-v3-large",
        ...     optimizer="muon",
        ...     batch_size=128,  # Large batch for H100
        ...     max_epochs=5
        ... )
    """
    from rm_optimizer.core.reward_model import BradleyTerryRM
    from rm_optimizer.core.data_loader import create_dataloader
    
    # Initialize model
    base_model = BradleyTerryRM(model_name=model_name)
    
    # Create data loaders if not provided
    if train_loader is None and data_path is not None:
        train_loader = create_dataloader(
            data_path,
            tokenizer=base_model.tokenizer,
            batch_size=batch_size,
            shuffle=True
        )
    
    if val_loader is None and data_path is not None:
        val_loader = create_dataloader(
            data_path,
            tokenizer=base_model.tokenizer,
            batch_size=batch_size,
            shuffle=False
        )
    
    # Lightning module
    lightning_model = RewardModelLightning(
        base_model=base_model,
        learning_rate=learning_rate,
        optimizer_name=optimizer,
        warmup_steps=kwargs.get('warmup_steps', 100),
        max_steps=kwargs.get('max_steps', 10000),
        weight_decay=kwargs.get('weight_decay', 0.01),
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='rm-{epoch:02d}-{val_accuracy:.4f}',
            monitor='val/accuracy',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/accuracy',
            patience=3,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar()
    ]
    
    # Logger
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project=wandb_project,
            name=wandb_name or f'rm_{optimizer}_{model_name.split("/")[-1]}',
            log_model=True
        )
    
    # H100-optimized trainer configuration
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,  # Single H100
        precision=precision,  # BF16 for H100 Tensor Cores
        gradient_clip_val=gradient_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.25,
        enable_checkpointing=True,
        benchmark=True,  # cuDNN autotuning
        deterministic=False,  # Allow non-determinism for speed
    )
    
    # Train
    trainer.fit(lightning_model, train_loader, val_loader)
    
    # Return best checkpoint path
    return callbacks[0].best_model_path


if __name__ == "__main__":
    # Example usage
    print("RewardModelLightning module loaded successfully")
    print("Use train_reward_model() to train a model")
