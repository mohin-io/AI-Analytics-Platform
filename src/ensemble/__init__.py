"""
Advanced Ensemble Methods Module

Implements sophisticated ensemble techniques beyond basic bagging and boosting.
"""

from src.ensemble.stacking import StackingEnsemble, MultiLevelStacking
from src.ensemble.blending import BlendingEnsemble
from src.ensemble.voting import WeightedVotingEnsemble, AdaptiveVoting

__all__ = [
    'StackingEnsemble',
    'MultiLevelStacking',
    'BlendingEnsemble',
    'WeightedVotingEnsemble',
    'AdaptiveVoting'
]
