"""Data efficiency module exports."""

from rm_optimizer.data_efficiency.active_learning import (
    ActiveLearner,
    UncertaintySampling,
    ExpectedModelChange,
    QueryByCommittee
)
from rm_optimizer.data_efficiency.margin_analysis import (
    MarginAnalyzer,
    CurriculumSampler
)

__all__ = [
    "ActiveLearner",
    "UncertaintySampling",
    "ExpectedModelChange",
    "QueryByCommittee",
    "MarginAnalyzer",
    "CurriculumSampler",
]
