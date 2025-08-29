"""
RCFT Metrics - Utility functions for analyzing RCFT experiments
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score

def compute_ccdi(correlation, coherence):
    """
    Compute Coherence-Correlation Divergence Index (CCDI)
    
    Parameters:
    -----------
    correlation : float
        Correlation between initial and current state
    coherence : float
        Coherence (inverse variance) of current state
        
    Returns:
    --------
    float
        CCDI value (coherence - correlation)
    """
    return coherence - correlation


def classify_recovery(correlation_curve, perturbation_steps, threshold=0.5):
    """
    Classify recovery trajectory as true, false, or oscillatory
    
    Parameters:
    -----------
    correlation_curve : array-like
        Time series of correlation values
    perturbation_steps : list
        Indices of perturbation events
    threshold : float
        Threshold for correlation recovery quality
    
    Returns:
    --------
    str
        Classification of recovery ('true', 'false', 'oscillatory', or 'unknown')
    """
    # Extract correlation after second perturbation
    second_perturbation = perturbation_steps[1] if len(perturbation_steps) > 1 else perturbation_steps[0]
    post_perturb = correlation_curve[second_perturbation:]
    
    if len(post_perturb) < 3:
        return "unknown"  # Not enough data
    
    # Find peaks and valleys
    peaks = []
    valleys = []
    
    for i in range(1, len(post_perturb)-1):
        if post_perturb[i] > post_perturb[i-1] and post_perturb[i] > post_perturb[i+1]:
            peaks.append(i)
        if post_perturb[i] < post_perturb[i-1] and post_perturb[i] < post_perturb[i+1]:
            valleys.append(i)
    
    # Analyze pattern
    if len(peaks) == 0:
        # No peaks - check if monotonically increasing or decreasing
        if post_perturb[-1] > post_perturb[0]:
            return "true"  # Monotonic recovery
        else:
            return "false"  # Monotonic decline
    
    elif len(peaks) == 1:
        # One peak - check for peak-then-decline
        peak_idx = peaks[0]
        if peak_idx < len(post_perturb) // 2 and post_perturb[-1] < post_perturb[peak_idx]:
            return "false"  # Peak-then-decline
        else:
            return "true"  # Still recovering
    
    else:
        # Multiple peaks and valleys - oscillatory
        return "oscillatory"


def extract_attractor_residual(final_state, initial_state):
    """
    Compute the difference between final and initial states (Δ field)
    
    Parameters:
    -----------
    final_state : ndarray
        Final state of the field
    initial_state : ndarray
        Initial state of the field
    
    Returns:
    --------
    ndarray
        Residual field (final - initial)
    """
    return final_state - initial_state


def compute_mutual_information(state1, state2, bins=10):
    """
    Compute mutual information between two field states
    
    Parameters:
    -----------
    state1 : ndarray
        First field state
    state2 : ndarray
        Second field state
    bins : int
        Number of bins for discretization
    
    Returns:
    --------
    float
        Mutual information value
    """
    # Discretize the states
    state1_flat = state1.flatten()
    state2_flat = state2.flatten()
    
    # Scale to [0, bins-1] for binning
    state1_bins = np.floor(bins * (state1_flat + 1) / 2).astype(int)
    state1_bins = np.clip(state1_bins, 0, bins-1)
    
    state2_bins = np.floor(bins * (state2_flat + 1) / 2).astype(int)
    state2_bins = np.clip(state2_bins, 0, bins-1)
    
    # Compute mutual information
    mi = mutual_info_score(state1_bins, state2_bins)
    return mi


def compute_spectral_entropy(state):
    """
    Compute spectral entropy of a field state
    
    Parameters:
    -----------
    state : ndarray
        Field state
    
    Returns:
    --------
    float
        Spectral entropy value
    """
    # Compute FFT
    fft = np.abs(np.fft.fft2(state))
    
    # Normalize
    fft_norm = fft / np.sum(fft)
    
    # Compute entropy
    entropy = -np.sum(fft_norm * np.log2(fft_norm + 1e-10))
    return entropy


def compute_recovery_metrics(initial_state, final_state, perturbation_step, correlation_curve):
    """
    Compute comprehensive recovery metrics
    
    Parameters:
    -----------
    initial_state : ndarray
        Initial field state
    final_state : ndarray
        Final field state
    perturbation_step : int
        Time step when perturbation was applied
    correlation_curve : array-like
        Time series of correlation values
    
    Returns:
    --------
    dict
        Dictionary of recovery metrics
    """
    # Correlation recovery
    initial_correlation = 1.0  # By definition
    perturb_correlation = correlation_curve[perturbation_step]
    final_correlation = correlation_curve[-1]
    
    # Calculate recovery quality (0 to 1 scale)
    if initial_correlation == perturb_correlation:
        recovery_correlation = 1.0
    else:
        recovery_correlation = (final_correlation - perturb_correlation) / (initial_correlation - perturb_correlation)
        recovery_correlation = min(max(recovery_correlation, 0), 1)  # Clamp to [0,1]
    
    # Compute mutual information
    final_mi = compute_mutual_information(initial_state, final_state)
    
    # Compute spectral entropy
    initial_entropy = compute_spectral_entropy(initial_state)
    final_entropy = compute_spectral_entropy(final_state)
    entropy_delta = final_entropy - initial_entropy
    
    # Compute residual norm
    residual = extract_attractor_residual(final_state, initial_state)
    residual_norm = np.linalg.norm(residual)
    
    # Return comprehensive metrics
    return {
        'final_correlation': final_correlation,
        'recovery_correlation': recovery_correlation,
        'mutual_information': final_mi,
        'initial_entropy': initial_entropy,
        'final_entropy': final_entropy,
        'entropy_delta': entropy_delta,
        'residual_norm': residual_norm
    }


def compute_field_statistics(state):
    """
    Compute statistical properties of a field state
    
    Parameters:
    -----------
    state : ndarray
        Field state
    
    Returns:
    --------
    dict
        Dictionary of statistical properties
    """
    # Flatten for statistical analysis
    flat_state = state.flatten()
    
    # Basic statistics
    mean = np.mean(flat_state)
    std = np.std(flat_state)
    min_val = np.min(flat_state)
    max_val = np.max(flat_state)
    
    # Compute higher moments
    skewness = stats.skew(flat_state)
    kurtosis = stats.kurtosis(flat_state)
    
    # Return statistics
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'entropy': compute_spectral_entropy(state)
    }