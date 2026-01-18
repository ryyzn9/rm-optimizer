"""
Unit tests for core module.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch


class TestPreferencePair:
    """Tests for PreferencePair dataclass."""
    
    def test_valid_creation(self):
        """Test creating a valid preference pair."""
        from rm_optimizer.core.base import PreferencePair
        
        pair = PreferencePair(
            prompt="What is 2+2?",
            chosen="The answer is 4.",
            rejected="The answer is 5."
        )
        
        assert pair.prompt == "What is 2+2?"
        assert pair.chosen == "The answer is 4."
        assert pair.rejected == "The answer is 5."
        assert pair.margin is None
    
    def test_with_margin(self):
        """Test preference pair with margin."""
        from rm_optimizer.core.base import PreferencePair
        
        pair = PreferencePair(
            prompt="Test",
            chosen="Good",
            rejected="Bad",
            margin=0.8
        )
        
        assert pair.margin == 0.8
    
    def test_empty_prompt_raises(self):
        """Test that empty prompt raises error."""
        from rm_optimizer.core.base import PreferencePair
        
        with pytest.raises(ValueError, match="non-empty"):
            PreferencePair(prompt="", chosen="Good", rejected="Bad")
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        from rm_optimizer.core.base import PreferencePair
        
        pair = PreferencePair(
            prompt="Test",
            chosen="Good",
            rejected="Bad",
            margin=0.5
        )
        
        d = pair.to_dict()
        assert d['prompt'] == "Test"
        assert d['margin'] == 0.5
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        from rm_optimizer.core.base import PreferencePair
        
        data = {
            'prompt': 'Test',
            'chosen': 'Good',
            'rejected': 'Bad',
            'margin': 0.5
        }
        
        pair = PreferencePair.from_dict(data)
        assert pair.prompt == "Test"
        assert pair.margin == 0.5


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics."""
    
    def test_creation(self):
        """Test creating evaluation metrics."""
        from rm_optimizer.core.base import EvaluationMetrics
        
        metrics = EvaluationMetrics(
            accuracy=0.85,
            ece=0.05,
            brier=0.12
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.ece == 0.05
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        from rm_optimizer.core.base import EvaluationMetrics
        
        metrics = EvaluationMetrics(accuracy=0.9)
        d = metrics.to_dict()
        
        assert d['accuracy'] == 0.9
        assert 'ece' in d


class TestHessianSpectrum:
    """Tests for HessianSpectrum."""
    
    def test_creation(self):
        """Test creating Hessian spectrum."""
        from rm_optimizer.landscape.hessian import HessianSpectrum
        
        spectrum = HessianSpectrum(
            eigenvalues=np.array([100.0, 50.0, 25.0]),
            trace_estimate=175.0,
            condition_number=4.0,
            effective_rank=2.5,
            sharpness_score=0.57,
            flatness_index=0.01
        )
        
        assert spectrum.eigenvalues[0] == 100.0
        assert spectrum.trace_estimate == 175.0
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        from rm_optimizer.landscape.hessian import HessianSpectrum
        
        spectrum = HessianSpectrum(
            eigenvalues=np.array([100.0, 50.0]),
            trace_estimate=150.0,
            condition_number=2.0,
            effective_rank=1.8,
            sharpness_score=0.67,
            flatness_index=0.01
        )
        
        d = spectrum.to_dict()
        assert isinstance(d['eigenvalues'], list)
        assert d['trace_estimate'] == 150.0


class TestRLReadinessScorer:
    """Tests for RL-readiness scoring."""
    
    def test_high_score(self):
        """Test high RL-readiness score."""
        from rm_optimizer.rl_coupling.rl_readiness import RLReadinessScorer
        
        scorer = RLReadinessScorer()
        report = scorer.compute_score(
            accuracy=0.92,
            ece=0.03,
            reward_variance=0.5,
            top_eigenvalue=30.0
        )
        
        assert report.overall_score > 0.8
        assert "ready" in report.recommendations[0].lower()
    
    def test_low_score(self):
        """Test low RL-readiness score."""
        from rm_optimizer.rl_coupling.rl_readiness import RLReadinessScorer
        
        scorer = RLReadinessScorer()
        report = scorer.compute_score(
            accuracy=0.55,
            ece=0.25,
            reward_variance=3.0,
            top_eigenvalue=500.0
        )
        
        assert report.overall_score < 0.6
        assert "not recommended" in report.recommendations[0].lower()


class TestActivelearning:
    """Tests for active learning components."""
    
    def test_unlabeled_pair_creation(self):
        """Test creating unlabeled pair."""
        from rm_optimizer.data_efficiency.active_learning import UnlabeledPair
        
        pair = UnlabeledPair(
            prompt="Test",
            response_a="A",
            response_b="B"
        )
        
        assert pair.prompt == "Test"
        assert pair.acquisition_score == 0.0


class TestMarginAnalysis:
    """Tests for margin analysis."""
    
    def test_margin_statistics(self):
        """Test margin statistics computation."""
        from rm_optimizer.data_efficiency.margin_analysis import MarginAnalyzer, MarginStatistics
        
        analyzer = MarginAnalyzer()
        
        margins = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        stats = analyzer.analyze(margins)
        
        assert stats.mean == pytest.approx(0.5)
        assert stats.min == 0.1
        assert stats.max == 0.9
    
    def test_curriculum_sampler(self):
        """Test curriculum sampler."""
        from rm_optimizer.data_efficiency.margin_analysis import CurriculumSampler
        
        margins = np.array([0.1, 0.3, 0.5, 0.8, 0.9])
        sampler = CurriculumSampler(margins)
        
        samples = sampler.sample(10)
        assert len(samples) == 10
        assert all(0 <= s < len(margins) for s in samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
