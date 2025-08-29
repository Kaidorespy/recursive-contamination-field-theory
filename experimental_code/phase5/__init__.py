"""
Initialization file for Phase V modules.
"""

# Import modules
from .temporal_coherence import TemporalCoherenceReinforcer
from .self_distinction import SelfDistinctionAnalyzer
from .identity_biasing import IdentityBiaser
from .temporal_adjacency import TemporalAdjacencyEncoder
from .echo_stability import EchoTrailAnalyzer
from .module_base import PhaseVModule

__all__ = [
    'TemporalCoherenceReinforcer',
    'SelfDistinctionAnalyzer',
    'IdentityBiaser',
    'TemporalAdjacencyEncoder',
    'EchoTrailAnalyzer',
    'PhaseVModule'
]