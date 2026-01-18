"""
Concrete reward model implementations.

This module provides:
- BradleyTerryRM: Standard Bradley-Terry loss reward model

H100 Optimizations:
- FlashAttention-2 for memory-efficient attention
- BF16 precision for Tensor Core acceleration
- torch.compile() for 20-30% speedup
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rm_optimizer.core.base import BaseRewardModel


class BradleyTerryRM(BaseRewardModel):
    """
    Bradley-Terry reward model using HuggingFace transformers.
    
    The Bradley-Terry model assumes pairwise preferences follow
    a logistic distribution:
    
        P(A > B) = σ(r(A) - r(B))
    
    where σ is the sigmoid function and r is the reward.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to run on ('cuda' or 'cpu')
        max_length: Maximum sequence length
        compile_model: Whether to apply torch.compile()
        use_flash_attention: Whether to use FlashAttention-2
        dtype: Model dtype (torch.bfloat16 for H100)
    
    Example:
        >>> model = BradleyTerryRM("microsoft/deberta-v3-large")
        >>> margin = model.score_pair(
        ...     "What is 2+2?",
        ...     "The answer is 4.",
        ...     "The answer is 5."
        ... )
        >>> print(f"Margin: {margin:.3f}")  # Positive = first response preferred
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        device: str = "cuda",
        max_length: int = 512,
        compile_model: bool = True,
        use_flash_attention: bool = True,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            compile_model=compile_model,
            use_flash_attention=use_flash_attention
        )
        
        self.max_length = max_length
        self.dtype = dtype
        
        # Determine attention implementation
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        
        # Load pretrained model with classification head
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,  # Scalar reward output
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback without flash attention (for unsupported models)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        
        self.model = self.model.to(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply torch.compile() for H100 optimization
        if compile_model and device == "cuda":
            self.compile()
    
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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Squeeze from (batch_size, 1) to (batch_size,)
        return outputs.logits.squeeze(-1)
    
    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry loss for a batch.
        
        Loss = -log(σ(r_chosen - r_rejected))
        
        This is equivalent to binary cross-entropy where the
        label is always 1 (chosen is preferred).
        
        Args:
            chosen_ids: Token IDs for chosen responses
            chosen_mask: Attention mask for chosen responses
            rejected_ids: Token IDs for rejected responses
            rejected_mask: Attention mask for rejected responses
        
        Returns:
            loss: Scalar loss value
        """
        # Forward pass for both
        reward_chosen = self.forward(chosen_ids, chosen_mask)
        reward_rejected = self.forward(rejected_ids, rejected_mask)
        
        # Bradley-Terry loss: -log(sigmoid(margin))
        # Numerically stable implementation
        margin = reward_chosen - reward_rejected
        loss = -torch.nn.functional.logsigmoid(margin).mean()
        
        return loss
    
    def compute_loss_with_metrics(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss and additional metrics for logging.
        
        Returns:
            Dictionary with loss, accuracy, margin, reward statistics
        """
        reward_chosen = self.forward(chosen_ids, chosen_mask)
        reward_rejected = self.forward(rejected_ids, rejected_mask)
        
        margin = reward_chosen - reward_rejected
        loss = -torch.nn.functional.logsigmoid(margin).mean()
        
        # Accuracy: fraction where chosen reward > rejected reward
        accuracy = (margin > 0).float().mean()
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "margin_mean": margin.mean(),
            "margin_std": margin.std(),
            "reward_chosen_mean": reward_chosen.mean(),
            "reward_rejected_mean": reward_rejected.mean(),
        }
    
    def tokenize(
        self, 
        prompt: str, 
        response: str
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize prompt + response pair.
        
        Concatenates prompt and response with appropriate formatting.
        
        Args:
            prompt: Input prompt
            response: Response text
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Format: "Prompt: {prompt}\n\nResponse: {response}"
        text = f"Prompt: {prompt}\n\nResponse: {response}"
        
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    
    def tokenize_batch(
        self,
        prompts: list,
        responses: list
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of prompt-response pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses
        
        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'
        """
        texts = [
            f"Prompt: {p}\n\nResponse: {r}" 
            for p, r in zip(prompts, responses)
        ]
        
        tokens = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    
    def save_pretrained(self, path: str) -> None:
        """Save model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
        **kwargs
    ) -> "BradleyTerryRM":
        """Load model from disk."""
        instance = cls.__new__(cls)
        BaseRewardModel.__init__(
            instance,
            model_name=path,
            device=device,
            **kwargs
        )
        
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            torch_dtype=kwargs.get("dtype", torch.bfloat16),
        ).to(device)
        
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.max_length = kwargs.get("max_length", 512)
        
        return instance


class PlackettLuceRM(BaseRewardModel):
    """
    Plackett-Luce reward model for ranking > 2 items.
    
    Extends Bradley-Terry to handle rankings of multiple items.
    
    Note: Implementation placeholder - extend if needed for ranking tasks.
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PlackettLuceRM is not yet implemented. "
            "Use BradleyTerryRM for pairwise preference tasks."
        )
    
    def forward(self, input_ids, attention_mask):
        pass
    
    def compute_loss(self, *args, **kwargs):
        pass
    
    def tokenize(self, prompt, response):
        pass
