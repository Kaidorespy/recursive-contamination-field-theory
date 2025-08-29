"""
Metrics utilities for Phase V of the RCFT framework.
This module provides tools for calculating identity and memory metrics.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score

class IdentityMetrics:
    """Tools for calculating identity-specific metrics."""
    
    @staticmethod
    def correlation(state1, state2):
        """
        Compute correlation between two states.
        
        Args:
            state1: First state array
            state2: Second state array
            
        Returns:
            Correlation coefficient
        """
        flat1 = state1.flatten()
        flat2 = state2.flatten()
        
        # Use Pearson correlation coefficient
        corr, _ = stats.pearsonr(flat1, flat2)
        return corr
    
    @staticmethod
    def coherence(state):
        """
        Compute coherence (inverse of variance) of a state.
        
        Args:
            state: State array
            
        Returns:
            Coherence value
        """
        # Coherence is defined as 1/(1+variance)
        variance = np.var(state)
        coherence = 1 / (1 + variance)
        return coherence
    
    @staticmethod
    def ccdi(state, reference_state):
        """
        Compute Coherence-Correlation Divergence Index (CCDI).
        
        Args:
            state: Current state
            reference_state: Reference state
            
        Returns:
            CCDI value
        """
        # CCDI = coherence - correlation
        coherence = IdentityMetrics.coherence(state)
        correlation = IdentityMetrics.correlation(state, reference_state)
        
        return coherence - correlation
    
    @staticmethod
    def mutual_information(state1, state2, bins=10):
        """
        Compute mutual information between two states.
        
        Args:
            state1: First state array
            state2: Second state array
            bins: Number of bins for discretization
            
        Returns:
            Mutual information value
        """
        flat1 = state1.flatten()
        flat2 = state2.flatten()
        
        # Discretize the states
        bins1 = np.linspace(np.min(flat1), np.max(flat1), bins+1)
        bins2 = np.linspace(np.min(flat2), np.max(flat2), bins+1)
        
        binned1 = np.digitize(flat1, bins1)
        binned2 = np.digitize(flat2, bins2)
        
        # Calculate mutual information
        mi = mutual_info_score(binned1, binned2)
        
        # Normalize by the entropy of the original state
        entropy1 = stats.entropy(np.bincount(binned1))
        normalized_mi = mi / entropy1 if entropy1 > 0 else 0
        
        return normalized_mi
    
    @staticmethod
    def spectral_entropy(state):
        """
        Compute spectral entropy of a state using FFT.
        
        Args:
            state: State array
            
        Returns:
            Spectral entropy value
        """
        from scipy.fft import fft2
        
        # Compute 2D FFT
        fft_result = fft2(state)
        fft_mag = np.abs(fft_result)
        
        # Normalize by total power
        total_power = np.sum(fft_mag)
        
        if total_power > 0:
            normalized_fft = fft_mag / total_power
            
            # Calculate entropy (avoid log(0))
            nonzero = normalized_fft > 0
            entropy = -np.sum(normalized_fft[nonzero] * np.log2(normalized_fft[nonzero]))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(np.size(state))
            normalized_entropy = entropy / max_entropy
            
            return normalized_entropy
        else:
            return 0
    
    @staticmethod
    def spatial_entropy(state, bins=10):
        """
        Compute spatial entropy of a state.
        
        Args:
            state: State array
            bins: Number of bins for histogram
            
        Returns:
            Spatial entropy value
        """
        flat = state.flatten()
        
        # Create histogram
        hist, _ = np.histogram(flat, bins=bins, density=True)
        
        # Calculate entropy (avoid log(0))
        nonzero = hist > 0
        entropy = -np.sum(hist[nonzero] * np.log2(hist[nonzero]))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(bins)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    @staticmethod
    def attractor_stability(states):
        """
        Compute stability of an attractor based on state trajectory.
        
        Args:
            states: List of state arrays representing a trajectory
            
        Returns:
            Stability value (higher means more stable)
        """
        if len(states) < 2:
            return 1.0  # Single state is stable by default
            
        # Calculate correlations between consecutive states
        correlations = []
        for i in range(1, len(states)):
            corr = IdentityMetrics.correlation(states[i-1], states[i])
            correlations.append(corr)
            
        # Stability is measured by the consistency of correlations
        # High and consistent correlation = stable attractor
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        # Penalize both low correlation and high variance
        stability = mean_corr / (1 + std_corr)
        
        return stability
    
    @staticmethod
    def centroid_drift(states):
        """
        Compute drift of attractor centroid over time.
        
        Args:
            states: List of state arrays
            
        Returns:
            Drift distance and intermediate distances
        """
        if len(states) < 2:
            return 0.0, [0.0]
            
        # Calculate centroids by thresholding states
        centroids = []
        for state in states:
            # Threshold state to find active regions
            threshold = np.mean(state) + 0.5 * np.std(state)
            active = state > threshold
            
            if np.any(active):
                # Calculate weighted centroid
                y_indices, x_indices = np.indices(state.shape)
                weighted_x = np.sum(x_indices * active * state) / np.sum(active * state)
                weighted_y = np.sum(y_indices * active * state) / np.sum(active * state)
                centroids.append((weighted_x, weighted_y))
            else:
                # Fallback to geometric center if no active regions
                centroids.append((state.shape[1]/2, state.shape[0]/2))
                
        # Calculate distances between consecutive centroids
        distances = []
        for i in range(1, len(centroids)):
            dist = np.sqrt((centroids[i][0] - centroids[i-1][0])**2 + 
                         (centroids[i][1] - centroids[i-1][1])**2)
            distances.append(dist)
            
        # Total drift is sum of all distances
        total_drift = sum(distances)
        
        return total_drift, distances
    
    @staticmethod
    def temporal_coherence(states, window_size=5):
        """
        Compute temporal coherence across a sequence of states.
        
        Args:
            states: List of state arrays
            window_size: Size of the window for calculating local coherence
            
        Returns:
            Temporal coherence value and local coherence values
        """
        if len(states) < window_size:
            return 1.0, [1.0]  # Not enough states for temporal analysis
            
        # Calculate correlations within sliding windows
        local_coherence = []
        for i in range(len(states) - window_size + 1):
            window = states[i:i+window_size]
            
            # Calculate pairwise correlations within window
            correlations = []
            for j in range(window_size):
                for k in range(j+1, window_size):
                    corr = IdentityMetrics.correlation(window[j], window[k])
                    correlations.append(corr)
                    
            # Local coherence is mean of correlations
            local_coherence.append(np.mean(correlations))
            
        # Temporal coherence is mean of local coherence values
        temporal_coherence = np.mean(local_coherence)
        
        return temporal_coherence, local_coherence
    
    @staticmethod
    def self_distinction_index(state, echo_states, reference_state=None):
        """
        Compute self-distinction index (how well a state preserves identity).
        
        Args:
            state: Current state
            echo_states: Previous states (echoes)
            reference_state: Optional initial reference state
            
        Returns:
            Self-distinction index value
        """
        if not echo_states:
            return 1.0  # No echoes to compare against
            
        # If reference provided, include it in comparisons
        if reference_state is not None:
            states_to_compare = echo_states + [reference_state]
        else:
            states_to_compare = echo_states
            
        # Calculate correlations with all comparison states
        correlations = []
        for compare_state in states_to_compare:
            corr = IdentityMetrics.correlation(state, compare_state)
            correlations.append(corr)
            
        # Calculate coherence of current state
        coherence = IdentityMetrics.coherence(state)
        
        # Self-distinction index combines correlation with past states and current coherence
        # High correlation with past states and high coherence = strong self-identity
        mean_corr = np.mean(correlations)
        
        # Weight coherence and correlation equally
        self_distinction = 0.5 * mean_corr + 0.5 * coherence
        
        return self_distinction
    
    @staticmethod
    def identity_persistence(states, reference_state=None):
        """
        Compute identity persistence across a sequence of states.
        
        Args:
            states: List of state arrays
            reference_state: Optional reference state (if None, use first state)
            
        Returns:
            Identity persistence value
        """
        if len(states) < 2:
            return 1.0  # Single state has perfect persistence
            
        # Use first state as reference if none provided
        ref = reference_state if reference_state is not None else states[0]
        
        # Calculate correlation with reference for each state
        correlations = []
        for state in states:
            corr = IdentityMetrics.correlation(state, ref)
            correlations.append(corr)
            
        # Persistence is area under the correlation curve
        # Normalize by number of states
        persistence = np.trapz(correlations) / (len(states) - 1)
        
        return persistence
    
    @staticmethod
    def recovery_quality(initial_state, perturbed_state, final_state):
        """
        Compute quality of recovery from perturbation.
        
        Args:
            initial_state: State before perturbation
            perturbed_state: State after perturbation
            final_state: State after recovery
            
        Returns:
            Recovery quality value
        """
        # Calculate correlations
        initial_corr = IdentityMetrics.correlation(initial_state, perturbed_state)
        final_corr = IdentityMetrics.correlation(initial_state, final_state)
        
        # Recovery quality measures how much correlation improved
        # If initial correlation was already perfect, recovery quality is 1
        if initial_corr >= 0.99:
            return 1.0
            
        # Calculate how much of the perturbation was recovered
        recovery_quality = (final_corr - initial_corr) / (1 - initial_corr)
        
        # Clamp to [0, 1] range
        recovery_quality = max(0, min(1, recovery_quality))
        
        return recovery_quality
    
    @staticmethod
    def echo_correction_delta(original_recovery, echo_recovery):
        """
        Compute the improvement in recovery due to echo.
        
        Args:
            original_recovery: Recovery quality without echo
            echo_recovery: Recovery quality with echo
            
        Returns:
            Echo correction delta
        """
        # Simple difference in recovery quality
        delta = echo_recovery - original_recovery
        
        # Normalize by potential improvement
        potential = 1.0 - original_recovery
        if potential > 0:
            normalized_delta = delta / potential
        else:
            normalized_delta = 0.0 if delta == 0.0 else 1.0
            
        return normalized_delta
    
    @staticmethod
    def bias_success_rate(target_fingerprint, biased_fingerprint, unbiased_fingerprint):
        """
        Compute success rate of bias in steering toward target.
        
        Args:
            target_fingerprint: Target fingerprint (dictionary)
            biased_fingerprint: Fingerprint with bias applied
            unbiased_fingerprint: Fingerprint without bias
            
        Returns:
            Bias success rate
        """
        from .fingerprinting import AttractorFingerprinter
        
        # Create temporary fingerprinter
        fingerprinter = AttractorFingerprinter()
        
        # Calculate distances
        biased_dist = fingerprinter.compute_fingerprint_distance(
            target_fingerprint, biased_fingerprint)
        unbiased_dist = fingerprinter.compute_fingerprint_distance(
            target_fingerprint, unbiased_fingerprint)
        
        # Get aggregate distances
        biased_agg = biased_dist.get('aggregate_distance', 1.0)
        unbiased_agg = unbiased_dist.get('aggregate_distance', 1.0)
        
        # Success rate measures improvement in distance
        improvement = unbiased_agg - biased_agg
        
        # Normalize by starting distance
        if unbiased_agg > 0:
            success_rate = improvement / unbiased_agg
        else:
            success_rate = 1.0 if improvement >= 0 else 0.0
            
        # Clamp to [0, 1] range
        success_rate = max(0, min(1, success_rate))
        
        return success_rate
    
class IdentityTrace:
    """
    Tracks identity metrics for a state across iterations.
    """
    
    def __init__(self, initial_state, label=None):
        """
        Initialize identity trace.
        
        Args:
            initial_state: Initial state to track
            label: Optional label
        """
        self.states = [initial_state.copy()]
        self.label = label or "Identity Trace"
        
        # Metrics over time
        self.correlations = [1.0]  # Correlation with initial state
        self.coherence = [IdentityMetrics.coherence(initial_state)]
        self.ccdi = [0.0]  # CCDI starts at 0 (perfect correlation)
        self.spectral_entropy = [IdentityMetrics.spectral_entropy(initial_state)]
        self.spatial_entropy = [IdentityMetrics.spatial_entropy(initial_state)]
        
        # Iteration tracking
        self.iteration = 0
        
        # Fingerprints
        self.fingerprints = []
        
    def add_state(self, state, compute_fingerprint=True):
        """
        Add a new state to the trace.
        
        Args:
            state: New state
            compute_fingerprint: Whether to compute fingerprint
            
        Returns:
            Dictionary of updated metrics
        """
        self.iteration += 1
        self.states.append(state.copy())
        
        # Calculate metrics relative to initial state
        initial_state = self.states[0]
        
        correlation = IdentityMetrics.correlation(state, initial_state)
        self.correlations.append(correlation)
        
        coherence = IdentityMetrics.coherence(state)
        self.coherence.append(coherence)
        
        # CCDI
        ccdi = coherence - correlation
        self.ccdi.append(ccdi)
        
        # Entropy metrics
        spec_entropy = IdentityMetrics.spectral_entropy(state)
        self.spectral_entropy.append(spec_entropy)
        
        spat_entropy = IdentityMetrics.spatial_entropy(state)
        self.spatial_entropy.append(spat_entropy)
        
        # Compute fingerprint if requested
        if compute_fingerprint:
            from .fingerprinting import AttractorFingerprinter
            fingerprinter = AttractorFingerprinter()
            fingerprint = fingerprinter.compute_fingerprint(state)
            self.fingerprints.append(fingerprint)
            
        # Return updated metrics
        metrics = {
            'correlation': correlation,
            'coherence': coherence,
            'ccdi': ccdi,
            'spectral_entropy': spec_entropy,
            'spatial_entropy': spat_entropy,
            'iteration': self.iteration
        }
        
        return metrics
    
    def compute_identity_persistence(self):
        """
        Compute identity persistence across all states.
        
        Returns:
            Identity persistence value
        """
        return IdentityMetrics.identity_persistence(self.states)
    
    def compute_self_distinction_index(self):
        """
        Compute self-distinction index for the latest state.
        
        Returns:
            Self-distinction index value
        """
        if len(self.states) < 2:
            return 1.0
            
        current_state = self.states[-1]
        echo_states = self.states[:-1]
        
        return IdentityMetrics.self_distinction_index(current_state, echo_states)
    
    def compute_temporal_coherence(self, window_size=5):
        """
        Compute temporal coherence across states.
        
        Args:
            window_size: Size of window for local coherence
            
        Returns:
            Temporal coherence value
        """
        return IdentityMetrics.temporal_coherence(self.states, window_size)[0]
    
    def get_metrics_dict(self):
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary of metric arrays
        """
        return {
            'correlation': self.correlations,
            'coherence': self.coherence,
            'ccdi': self.ccdi,
            'spectral_entropy': self.spectral_entropy,
            'spatial_entropy': self.spatial_entropy,
            'identity_persistence': self.compute_identity_persistence(),
            'self_distinction': self.compute_self_distinction_index(),
            'temporal_coherence': self.compute_temporal_coherence()
        }
    
    def __repr__(self):
        """String representation of the trace."""
        return f"{self.label}: {self.iteration} iterations"
    
    def __str__(self):
        """Detailed string representation of the trace."""
        persistence = self.compute_identity_persistence()
        distinction = self.compute_self_distinction_index()
        
        return (
            f"{self.label}:\n"
            f"  Iterations: {self.iteration}\n"
            f"  Final correlation: {self.correlations[-1]:.4f}\n"
            f"  Final coherence: {self.coherence[-1]:.4f}\n"
            f"  Final CCDI: {self.ccdi[-1]:.4f}\n"
            f"  Identity persistence: {persistence:.4f}\n"
            f"  Self-distinction index: {distinction:.4f}"
        )