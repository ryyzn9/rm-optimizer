"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_preference_pairs():
    """Sample preference pairs for testing."""
    from rm_optimizer.core.base import PreferencePair
    
    return [
        PreferencePair(
            prompt="What is 2+2?",
            chosen="The answer is 4.",
            rejected="The answer is 5.",
            margin=0.8
        ),
        PreferencePair(
            prompt="Explain gravity.",
            chosen="Gravity is the force that attracts objects toward Earth.",
            rejected="Gravity makes things float.",
            margin=0.9
        ),
        PreferencePair(
            prompt="What color is the sky?",
            chosen="The sky is blue during the day.",
            rejected="The sky is always green.",
            margin=0.95
        )
    ]


@pytest.fixture
def sample_margins():
    """Sample margin values for testing."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


@pytest.fixture
def sample_eigenvalues():
    """Sample eigenvalues for testing."""
    return np.array([
        234.5, 198.2, 156.7, 123.4, 98.7,
        76.5, 54.3, 43.2, 32.1, 21.0,
        15.4, 12.3, 10.2, 8.1, 6.5,
        5.2, 4.1, 3.2, 2.5, 1.8
    ])
