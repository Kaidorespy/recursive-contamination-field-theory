#!/bin/bash
# Resonance Mapping — Drift-Lock vs Chaos

# Define parameters
SOURCE_CF="phase6_results/counterfactual/A_cfCF_5_intr0.90_after.npy"
BASE_DIR="phase7_results/ResonanceMapping"
STRENGTH=0.6
BLEND_RATIO=0.5
BLEND_METHOD="pixel"
STEPS=50

# Create output directories
mkdir -p $BASE_DIR

# Step 1: Create initial hybrid seed
echo "Creating initial hybrid seed..."
python run_phase7.py \
  --cf_sources $SOURCE_CF \
  --output_dir $BASE_DIR/A_7D1_seed \
  --blend_ratio $BLEND_RATIO \
  --blend_method $BLEND_METHOD \
  --strengths $STRENGTH \
  --steps $STEPS

# Extract the final state from the first run to use as R1 source
R1_SOURCE="$BASE_DIR/A_7D1_seed/injection_results/A_B_${BLEND_RATIO}_${BLEND_METHOD}_injected_CF_R1_source1_str${STRENGTH}_delay0_final.npy"

# Step 2: Recursive Application - Depth 1 (already done in Step 1)
echo "Recursive depth 1 completed in Step 1"

# Step 3: Recursive Application - Depth 2
echo "Running recursive depth 2..."
python run_phase7.py \
  --cf_sources $R1_SOURCE \
  --output_dir $BASE_DIR/A_7D1_R2 \
  --blend_ratio $BLEND_RATIO \
  --blend_method $BLEND_METHOD \
  --strengths $STRENGTH \
  --steps $STEPS

# Extract R2 final state
R2_SOURCE="$BASE_DIR/A_7D1_R2/injection_results/A_B_${BLEND_RATIO}_${BLEND_METHOD}_injected_CF_R1_source1_str${STRENGTH}_delay0_final.npy"

# Step 4: Recursive Application - Depth 4
echo "Running recursive depth 4..."
python run_phase7.py \
  --cf_sources $R2_SOURCE \
  --output_dir $BASE_DIR/A_7D1_R3 \
  --blend_ratio $BLEND_RATIO \
  --blend_method $BLEND_METHOD \
  --strengths $STRENGTH \
  --steps $STEPS

# Extract R3 final state
R3_SOURCE="$BASE_DIR/A_7D1_R3/injection_results/A_B_${BLEND_RATIO}_${BLEND_METHOD}_injected_CF_R1_source1_str${STRENGTH}_delay0_final.npy"

# Step 5: Recursive Application - Depth 6
echo "Running recursive depth 6..."
python run_phase7.py \
  --cf_sources $R3_SOURCE \
  --output_dir $BASE_DIR/A_7D1_R4 \
  --blend_ratio $BLEND_RATIO \
  --blend_method $BLEND_METHOD \
  --strengths $STRENGTH \
  --steps $STEPS

# Final step: Analyze drift across recursion depths
echo "Analyzing drift across recursion depths..."
python -c "
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# Define paths for each recursion level
base_dir = '$BASE_DIR'
seed_path = os.path.join(base_dir, 'A_7D1_seed/injection_results/A_B_${BLEND_RATIO}_${BLEND_METHOD}_injected_CF_R1_source1_str${STRENGTH}_delay0_final.npy')
r2_path = os.path.join(base_dir, 'A_7D1_R2/injection_results/A_B_${BLEND_RATIO}_${BLEND_METHOD}_injected_CF_R1_source1_str${STRENGTH}_delay0_final.npy')
r3_path = os.path.join(base_dir, 'A_7D1_R3/injection_results/A_B_${BLEND_RATIO}_${BLEND_METHOD}_injected_CF_R1_source1_str${STRENGTH}_delay0_final.npy')
r4_path = os.path.join(base_dir, 'A_7D1_R4/injection_results/A_B_${BLEND_RATIO}_${BLEND_METHOD}_injected_CF_R1_source1_str${STRENGTH}_delay0_final.npy')

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
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Correlation (drift from seed)
    plt.subplot(2, 2, 1)
    recursion_depths = [1, 2, 4, 6]
    correlations = [1.0, corr_r2, corr_r3, corr_r4]  # Seed corr with itself = 1.0
    plt.plot(recursion_depths, correlations, 'o-', linewidth=2)
    plt.title('Drift from Seed (Correlation)')
    plt.xlabel('Recursive Depth')
    plt.ylabel('Correlation with Seed')
    plt.grid(True)
    
    # Plot 2: Entropy
    plt.subplot(2, 2, 2)
    entropies = [entropy_seed, entropy_r2, entropy_r3, entropy_r4]
    plt.plot(recursion_depths, entropies, 'o-', linewidth=2, color='green')
    plt.title('Entropy vs Recursive Depth')
    plt.xlabel('Recursive Depth')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    # Plot 3: Skewness
    plt.subplot(2, 2, 3)
    skewness = [skew_seed, skew_r2, skew_r3, skew_r4]
    plt.plot(recursion_depths, skewness, 'o-', linewidth=2, color='red')
    plt.title('Skewness vs Recursive Depth')
    plt.xlabel('Recursive Depth')
    plt.ylabel('Skewness')
    plt.grid(True)
    
    # Plot 4: Summary with interpretation
    plt.subplot(2, 2, 4)
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
    
    summary_text = f'''
    RESONANCE MAPPING RESULTS
    
    System Behavior: {behavior}
    
    Drift trend: {drift_trend}
    Entropy trend: {entropy_trend}
    
    Final correlation with seed: {correlations[-1]:.4f}
    Final entropy: {entropies[-1]:.4f}
    
    Interpretation:
    {explanation}
    
    The {'increasing' if drift_trend == 'INCREASING' else 'stabilizing'} drift suggests 
    the system is {'diverging toward chaotic behavior' if drift_trend == 'INCREASING' else 'converging to a new attractor state'}
    '''
    
    plt.text(0.1, 0.1, summary_text, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'resonance_mapping_results.png'))
    
    # Also save the data as a CSV
    import pandas as pd
    results_df = pd.DataFrame({
        'Recursion_Depth': recursion_depths,
        'Correlation_With_Seed': correlations,
        'Entropy': entropies,
        'Skewness': skewness
    })
    results_df.to_csv(os.path.join(base_dir, 'resonance_mapping_data.csv'), index=False)
    
    print('\\nRESULTS SUMMARY:')
    print(f'System Behavior: {behavior}')
    print(f'Drift from seed: initial=0.0, final={1.0-correlations[-1]:.4f}')
    print(f'Entropy change: {entropies[-1] - entropies[0]:.4f}')
    print(f'Analysis complete. Results saved to {base_dir}/resonance_mapping_results.png')
    
except Exception as e:
    print(f'Error in analysis: {e}')
"

echo "Resonance Mapping experiment completed."