"""
Efficient Hessian eigenvalue computation for large models.

This module provides:
- HessianSpectrum: Results dataclass
- HessianAnalyzer: Compute Hessian eigenspectrum

Three computation methods:
1. PyHessian (Lanczos algorithm) - recommended
2. Power iteration (custom implementation)
3. Full eigendecomposition (small models only)

H100 Optimization:
- Uses TF32 for faster matmul in Hessian-vector products
- Large batch HVP computation (32 vs 8 on T4)
- ~5 min for top-50 eigenvalues on 7B model
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import time


@dataclass
class HessianSpectrum:
    """
    Results from Hessian eigenspectrum analysis.
    
    Attributes:
        eigenvalues: Top-k eigenvalues in descending order
        trace_estimate: Estimated trace of Hessian
        condition_number: λ_max / λ_min (measures ill-conditioning)
        effective_rank: (Σλ)² / Σλ² (intrinsic dimensionality)
        sharpness_score: λ_max / trace
        flatness_index: 1 / λ_max (inverse sharpness)
        model_name: Name of analyzed model
        optimizer_name: Optimizer used for training
        num_parameters: Number of model parameters
        computation_time: Time taken for analysis
    """
    eigenvalues: np.ndarray
    trace_estimate: float
    condition_number: float
    effective_rank: float
    sharpness_score: float
    flatness_index: float
    model_name: str = "unknown"
    optimizer_name: str = "unknown"
    num_parameters: int = 0
    computation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['eigenvalues'] = self.eigenvalues.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HessianSpectrum":
        """Create from dictionary."""
        data['eigenvalues'] = np.array(data['eigenvalues'])
        return cls(**data)
    
    def __str__(self) -> str:
        return (
            f"HessianSpectrum(\n"
            f"  top_eigenvalue={self.eigenvalues[0]:.4f},\n"
            f"  trace={self.trace_estimate:.4f},\n"
            f"  condition_number={self.condition_number:.4f},\n"
            f"  effective_rank={self.effective_rank:.4f},\n"
            f"  sharpness={self.sharpness_score:.6f},\n"
            f"  flatness={self.flatness_index:.6f}\n"
            f")"
        )


class HessianAnalyzer:
    """
    Compute Hessian eigenspectrum for reward models.
    
    For large models, we cannot materialize the full Hessian matrix
    (7B × 7B × 4 bytes = 196 PB). Instead, we use:
    
    1. Hessian-vector products (HVP): H·v = ∇(∇L·v)
    2. Lanczos/power iteration for top eigenvalues
    3. Hutchinson estimator for trace
    
    Args:
        model: PyTorch model to analyze
        device: Device for computation
        hvp_batch_size: Batch size for HVP (H100: 32, T4: 8)
    
    Example:
        >>> analyzer = HessianAnalyzer(model)
        >>> spectrum = analyzer.compute_spectrum(dataloader, top_k=50)
        >>> print(f"Top eigenvalue: {spectrum.eigenvalues[0]:.4f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        hvp_batch_size: int = 32  # H100 optimized
    ):
        self.model = model
        self.device = device
        self.hvp_batch_size = hvp_batch_size
        
        # Enable TF32 for faster Hessian computation on H100
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def compute_spectrum(
        self,
        dataloader,
        method: str = "pyhessian",
        top_k: int = 20,
        trace_samples: int = 100
    ) -> HessianSpectrum:
        """
        Compute Hessian eigenspectrum.
        
        Args:
            dataloader: DataLoader for Hessian computation
            method: 'pyhessian', 'power_iteration', or 'full'
            top_k: Number of top eigenvalues to compute
            trace_samples: Samples for trace estimation
        
        Returns:
            HessianSpectrum with results
        """
        start_time = time.time()
        self.model.eval()
        
        if method == "pyhessian":
            eigenvalues, trace = self._compute_pyhessian(dataloader, top_k, trace_samples)
        elif method == "power_iteration":
            eigenvalues, trace = self._compute_power_iteration(dataloader, top_k, trace_samples)
        elif method == "full":
            eigenvalues, trace = self._compute_full(dataloader)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute derived metrics
        eigenvalues = np.array(sorted(eigenvalues, reverse=True))
        
        # Handle edge cases
        if len(eigenvalues) == 0:
            eigenvalues = np.array([0.0])
        
        min_eigenval = eigenvalues[-1] if eigenvalues[-1] != 0 else 1e-10
        condition_number = eigenvalues[0] / abs(min_eigenval)
        
        eigenval_sum = eigenvalues.sum()
        eigenval_sq_sum = (eigenvalues ** 2).sum()
        effective_rank = (eigenval_sum ** 2) / eigenval_sq_sum if eigenval_sq_sum > 0 else 1.0
        
        sharpness_score = eigenvalues[0] / trace if trace > 0 else 0.0
        flatness_index = 1.0 / eigenvalues[0] if eigenvalues[0] > 0 else float('inf')
        
        elapsed = time.time() - start_time
        
        return HessianSpectrum(
            eigenvalues=eigenvalues,
            trace_estimate=trace,
            condition_number=condition_number,
            effective_rank=effective_rank,
            sharpness_score=sharpness_score,
            flatness_index=flatness_index,
            model_name=getattr(self.model, 'model_name', 'unknown'),
            optimizer_name='unknown',
            num_parameters=sum(p.numel() for p in self.model.parameters()),
            computation_time=elapsed
        )
    
    def _compute_pyhessian(
        self,
        dataloader,
        top_k: int,
        trace_samples: int
    ) -> Tuple[List[float], float]:
        """
        Use PyHessian library (recommended).
        
        PyHessian implements optimized Lanczos algorithm.
        """
        try:
            from pyhessian import hessian
        except ImportError:
            print("PyHessian not available, falling back to power iteration")
            return self._compute_power_iteration(dataloader, top_k, trace_samples)
        
        # Get a batch of data
        batch = next(iter(dataloader))
        
        # Move to device
        inputs = (
            batch['chosen_input_ids'].to(self.device),
            batch['chosen_attention_mask'].to(self.device),
            batch['rejected_input_ids'].to(self.device),
            batch['rejected_attention_mask'].to(self.device)
        )
        
        # Define loss function
        def loss_fn(*args):
            return self.model.compute_loss(*inputs)
        
        # Create Hessian computer
        hessian_comp = hessian(
            self.model,
            loss_fn,
            data=inputs,
            cuda=(self.device == "cuda")
        )
        
        # Compute top eigenvalues
        eigenvalues = hessian_comp.eigenvalues(top_n=top_k)
        
        # Estimate trace
        trace = hessian_comp.trace(maxIter=trace_samples)
        
        return eigenvalues, trace
    
    def _compute_power_iteration(
        self,
        dataloader,
        top_k: int,
        trace_samples: int
    ) -> Tuple[List[float], float]:
        """
        Custom power iteration implementation.
        
        Power iteration finds largest eigenvalue iteratively:
        v_{k+1} = H·v_k / ||H·v_k||
        λ = v^T H v
        """
        eigenvalues = []
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Compute first k eigenvalues via deflation
        prev_vectors = []
        
        for i in range(top_k):
            # Initialize random vector
            v = [torch.randn_like(p, device=self.device) for p in params]
            v = self._normalize(v)
            
            # Power iteration
            for iteration in range(50):
                # Compute Hessian-vector product
                hvp = self._hessian_vector_product(dataloader, params, v)
                
                # Deflate: remove components from previous eigenvectors
                for prev_v in prev_vectors:
                    proj = sum((pv * hv).sum() for pv, hv in zip(prev_v, hvp))
                    hvp = [hv - proj * pv for hv, pv in zip(hvp, prev_v)]
                
                # Normalize
                v_new = self._normalize(hvp)
                
                # Check convergence
                dot = abs(sum((vi * vn).sum() for vi, vn in zip(v, v_new)).item())
                if dot > 0.9999:
                    break
                
                v = v_new
            
            # Compute eigenvalue
            hvp = self._hessian_vector_product(dataloader, params, v)
            eigenvalue = sum((vi * hvi).sum() for vi, hvi in zip(v, hvp)).item()
            eigenvalues.append(eigenvalue)
            
            prev_vectors.append(v)
            
            if (i + 1) % 5 == 0:
                print(f"  Computed {i+1}/{top_k} eigenvalues")
        
        # Estimate trace
        trace = self._estimate_trace(dataloader, params, trace_samples)
        
        return eigenvalues, trace
    
    def _hessian_vector_product(
        self,
        dataloader,
        params: List[torch.Tensor],
        vector: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute H·v without forming H.
        
        Uses the identity: H·v = ∇(∇L·v)
        Two backward passes through automatic differentiation.
        """
        self.model.zero_grad()
        
        # Compute loss on a batch
        total_loss = 0.0
        count = 0
        
        for batch in dataloader:
            chosen_ids = batch['chosen_input_ids'].to(self.device)
            chosen_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_ids = batch['rejected_input_ids'].to(self.device)
            rejected_mask = batch['rejected_attention_mask'].to(self.device)
            
            loss = self.model.compute_loss(
                chosen_ids, chosen_mask,
                rejected_ids, rejected_mask
            )
            total_loss += loss
            count += 1
            
            if count >= self.hvp_batch_size // batch['chosen_input_ids'].shape[0]:
                break
        
        avg_loss = total_loss / count
        
        # First backward: compute gradients (keep graph)
        grads = torch.autograd.grad(
            avg_loss,
            params,
            create_graph=True,
            retain_graph=True
        )
        
        # Dot product: ∇L · v
        grad_vector_dot = sum((g * v).sum() for g, v in zip(grads, vector))
        
        # Second backward: compute ∇(∇L · v) = H·v
        hvp = torch.autograd.grad(
            grad_vector_dot,
            params,
            retain_graph=False
        )
        
        return [h.detach() for h in hvp]
    
    def _normalize(self, vectors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize vector to unit length."""
        norm = torch.sqrt(sum((v ** 2).sum() for v in vectors))
        if norm < 1e-10:
            norm = 1e-10
        return [v / norm for v in vectors]
    
    def _estimate_trace(
        self,
        dataloader,
        params: List[torch.Tensor],
        num_samples: int
    ) -> float:
        """
        Hutchinson trace estimator.
        
        tr(H) ≈ (1/m) Σ z^T H z, where z ~ N(0, I)
        """
        trace_estimate = 0.0
        
        for i in range(num_samples):
            # Random Gaussian vector
            z = [torch.randn_like(p, device=self.device) for p in params]
            
            # Compute H·z
            hz = self._hessian_vector_product(dataloader, params, z)
            
            # z^T H z
            trace_estimate += sum((zi * hzi).sum() for zi, hzi in zip(z, hz)).item()
            
            if (i + 1) % 20 == 0:
                print(f"  Trace estimation: {i+1}/{num_samples}")
        
        return trace_estimate / num_samples
    
    def _compute_full(self, dataloader) -> Tuple[List[float], float]:
        """
        Full Hessian eigendecomposition (small models only).
        
        WARNING: Memory O(d²), time O(d³)
        Only works for d < 10M parameters.
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        if num_params > 10_000_000:
            raise ValueError(
                f"Full Hessian infeasible for {num_params:,} params. "
                "Use 'pyhessian' or 'power_iteration'."
            )
        
        print(f"Computing full Hessian for {num_params:,} parameters...")
        print("This may take a while...")
        
        # Flatten parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Compute Hessian row by row
        hessian_rows = []
        
        for batch in dataloader:
            chosen_ids = batch['chosen_input_ids'].to(self.device)
            chosen_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_ids = batch['rejected_input_ids'].to(self.device)
            rejected_mask = batch['rejected_attention_mask'].to(self.device)
            
            loss = self.model.compute_loss(
                chosen_ids, chosen_mask,
                rejected_ids, rejected_mask
            )
            break
        
        # First order gradients
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grads_flat = torch.cat([g.reshape(-1) for g in grads])
        
        # Second order (Hessian rows)
        for i, g in enumerate(grads_flat):
            if i % 1000 == 0:
                print(f"  Row {i}/{len(grads_flat)}")
            row = torch.autograd.grad(g, params, retain_graph=True)
            row_flat = torch.cat([r.reshape(-1) for r in row])
            hessian_rows.append(row_flat.detach().cpu())
        
        hessian_matrix = torch.stack(hessian_rows)
        
        # Eigendecomposition
        eigenvalues, _ = torch.linalg.eigh(hessian_matrix)
        eigenvalues = eigenvalues.numpy()
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        trace = float(np.trace(hessian_matrix.numpy()))
        
        return eigenvalues.tolist(), trace
