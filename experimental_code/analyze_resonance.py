import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import pandas as pd

# Define paths to the files
seed_path = 'phase7_results/ResonanceMapping/A_7D1_seed/injection_results/A_B_0.50_pixel_injected_CF_R1_source1_str0.60_delay0_final.npy'
r2_path = 'phase7_results/DirectApproach/step2/injection_results/A_B_0.50_pixel_injected_CF_R1_source1_str0.60_delay0_final.npy'
r3_path = 'phase7_results/DirectApproach/step3/injection_results/A_B_0.50_pixel_injected_CF_R1_source1_str0.60_delay0_final.npy'
r4_path = 'phase7_results/DirectApproach/step4/injection_results/A_B_0.50_pixel_injected_CF_R1_source1_str0.60_delay0_final.npy'
output_dir = 'phase7_results/DirectApproach'

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Load states
try:
    seed_state = np.load(seed_path)
    r2_state = np.load(r2_path)
    r3_state = np.load(r3_path)
    r4_state = np.load(r4_path)
    
    # Calculate correlations (drift from seed)
    corr_r2 = np.corrcoef(seed_state.flatten(), r2_state.flatten())[0, 1]
    corr_r3 = np.corrcoef(seed_state.flatten(), r3_state.flatten())[0, 1]
    corr_r4 = np.corrcoef(seed_state.flatten(), r4_state.flatten())[0, 1]
    
    # Calculate sequential correlations
    seq_corr_r1_r2 = np.corrcoef(seed_state.flatten(), r2_state.flatten())[0, 1]
    seq_corr_r2_r3 = np.corrcoef(r2_state.flatten(), r3_state.flatten())[0, 1]
    seq_corr_r3_r4 = np.corrcoef(r3_state.flatten(), r4_state.flatten())[0, 1]
    
    # Calculate entropy and skewness
    entropy_seed = stats.entropy(np.histogram(seed_state, bins=100)[0])
    entropy_r2 = stats.entropy(np.histogram(r2_state, bins=100)[0])
    entropy_r3 = stats.entropy(np.histogram(r3_state, bins=100)[0])
    entropy_r4 = stats.entropy(np.histogram(r4_state, bins=100)[0])
    
    skew_seed = stats.skew(seed_state.flatten())
    skew_r2 = stats.skew(r2_state.flatten())
    skew_r3 = stats.skew(r3_state.flatten())
    skew_r4 = stats.skew(r4_state.flatten())
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Correlation (drift from seed)
    plt.subplot(3, 2, 1)
    recursion_depths = [1, 2, 4, 6]
    correlations = [1.0, corr_r2, corr_r3, corr_r4]  # Seed corr with itself = 1.0
    plt.plot(recursion_depths, correlations, 'o-', linewidth=2)
    plt.title('Drift from Seed (Correlation)')
    plt.xlabel('Recursive Depth')
    plt.ylabel('Correlation with Seed')
    plt.grid(True)
    
    # Plot 2: Sequential correlation
    plt.subplot(3, 2, 2)
    seq_depths = [1, 2, 3]
    seq_correlations = [seq_corr_r1_r2, seq_corr_r2_r3, seq_corr_r3_r4]
    plt.plot(seq_depths, seq_correlations, 'o-', linewidth=2, color='orange')
    plt.title('Sequential Correlations')
    plt.xlabel('Generation Pair (n → n+1)')
    plt.ylabel('Correlation')
    plt.grid(True)
    
    # Plot 3: Entropy
    plt.subplot(3, 2, 3)
    entropies = [entropy_seed, entropy_r2, entropy_r3, entropy_r4]
    plt.plot(recursion_depths, entropies, 'o-', linewidth=2, color='green')
    plt.title('Entropy vs Recursive Depth')
    plt.xlabel('Recursive Depth')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    # Plot 4: Skewness
    plt.subplot(3, 2, 4)
    skewness = [skew_seed, skew_r2, skew_r3, skew_r4]
    plt.plot(recursion_depths, skewness, 'o-', linewidth=2, color='red')
    plt.title('Skewness vs Recursive Depth')
    plt.xlabel('Recursive Depth')
    plt.ylabel('Skewness')
    plt.grid(True)
    
    # Plot 5: State Comparison
    plt.subplot(3, 2, 5)
    plt.imshow(np.vstack([
        np.hstack([seed_state, r2_state]),
        np.hstack([r3_state, r4_state])
    ]), cmap='viridis', vmin=-1, vmax=1)
    plt.title('State Evolution Across Recursions')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Plot 6: Summary with interpretation
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    # Determine if we see resonance or chaos
    if min(correlations) > 0.7:
        behavior = 'STABLE RESONANCE'
        explanation = 'The system maintains high correlation with seed state across recursions'
    elif correlations[-1] < 0.3:
        behavior = 'CHAOTIC INSTABILITY'
        explanation = 'The system diverges substantially from the seed state'
    elif correlations[-1] > correlations[-2]:
        behavior = 'CYCLIC RESONANCE'
        explanation = 'The system shows signs of returning toward the seed state'
    else:
        behavior = 'DRIFT WITH PLATEAU'
        explanation = 'The system appears to be stabilizing at an intermediate state'
    
    drift_trend = 'INCREASING' if correlations[-1] < correlations[-2] else 'PLATEAUING or DECREASING'
    entropy_trend = 'INCREASING' if entropies[-1] > entropies[-2] else 'PLATEAUING or DECREASING'
    sequential_trend = 'INCREASING' if seq_correlations[-1] > seq_correlations[-2] else 'DECREASING'
    
    summary_text = f'''
    RESONANCE MAPPING RESULTS
    
    System Behavior: {behavior}
    
    Drift from seed trend: {drift_trend}
    Sequential correlation trend: {sequential_trend}
    Entropy trend: {entropy_trend}
    
    Final correlation with seed: {correlations[-1]:.4f}
    Final entropy: {entropies[-1]:.4f}
    
    Interpretation:
    {explanation}
    
    The {'increasing' if drift_trend == 'INCREASING' else 'stabilizing'} drift suggests 
    the system is {'diverging toward chaotic behavior' if drift_trend == 'INCREASING' else 'converging to a new attractor state'}
    
    Sequential correlation trend suggests
    {'formation of a chaotic pattern' if sequential_trend == 'DECREASING' else 'emergence of a stable attractor'}
    '''
    
    plt.text(0.1, 0.1, summary_text, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resonance_mapping_results.png'))
    
    # Also save the data as a CSV
    results_df = pd.DataFrame({
        'Recursion_Depth': recursion_depths,
        'Correlation_With_Seed': correlations,
        'Entropy': entropies,
        'Skewness': skewness
    })
    results_df.to_csv(os.path.join(output_dir, 'resonance_mapping_data.csv'), index=False)
    
    # Add sequential correlation data
    seq_df = pd.DataFrame({
        'Generation_Pair': seq_depths,
        'Sequential_Correlation': seq_correlations
    })
    seq_df.to_csv(os.path.join(output_dir, 'sequential_correlation_data.csv'), index=False)
    
    print('\nRESULTS SUMMARY:')
    print(f'System Behavior: {behavior}')
    print(f'Drift from seed: initial=0.0, final={1.0-correlations[-1]:.4f}')
    print(f'Entropy change: {entropies[-1] - entropies[0]:.4f}')
    print(f'Sequential correlation trend: {sequential_trend}')
    print(f'Analysis complete. Results saved to {output_dir}/resonance_mapping_results.png')
    
except Exception as e:
    import traceback
    print(f'Error in analysis: {e}')
    traceback.print_exc()