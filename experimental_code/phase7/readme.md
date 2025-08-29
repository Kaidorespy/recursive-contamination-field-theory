# RCFT Phase VII: Self-Reflexive Contamination

This phase explores the recursive effects of memory contamination, examining what happens when corrupted memories are themselves used as sources for further contamination.

## Core Research Question

What happens when corrupted memories are re-used as counterfactuals? Can a system learn to replicate its own memory failures — and under what conditions does this recursive contamination stabilize, collapse, or self-repair?

## Conceptual Premise

Phase VII introduces recursive counterfactual exposure — using outputs from prior contamination phases as new counterfactuals (CF_R). These second-order CFs are injected into fresh hybrids to observe:

- Whether contamination amplifies or plateaus
- If prior failure biases future memory
- When recursive loops stabilize vs destabilize

## Components

The implementation consists of several key components:

1. **RecursiveContaminationEngine**: The main engine that handles loading prior memory states as recursive CFs, creating fresh hybrids, and injecting recursive CFs with tracking of memory lineage.

2. **RecursiveMemoryAnalyzer**: Utility class for analyzing recursive memory patterns and drift, including calculation of the Recursive Fragility Index (RFI).

## Running Experiments

To run a Phase VII experiment:

```bash
python run_phase7.py --output_dir phase7_results --phase6_dir phase6_results
```

The script will:
1. Load recursive CF sources from Phase 6 outputs (or use manually specified sources)
2. Create a fresh hybrid from two base patterns
3. Inject the recursive CFs at different strengths
4. Analyze the results and create visualizations

## Advanced Options

```bash
python run_phase7.py --cf_sources path1.npy,path2.npy --strengths 0.4,0.6,0.8 --blend_ratio 0.5 --blend_method pixel --delay 0 --steps 50
```

## Key Metrics

- **Memory Integrity Delta**: How much the original memory is corrupted
- **CF Influence**: How much the final state correlates with the recursive CF
- **Recursive Drift**: How far the final state drifts from the pre-injection state
- **Recovery Bias**: Bias toward pattern A vs pattern B in the final state
- **Recursive Fragility Index (RFI)**: The rate at which a system forgets its origins over recursive injections
- **Attractor Melting**: Whether the final state fails to correlate with any known attractor

## Visualizations

The system generates various visualizations:
- Detailed state visualizations for each CF and injection
- Summary plots showing metrics by injection strength and CF
- Memory lineage maps tracking the relationships between recursive generations
- CF space projections using dimensionality reduction

## Recursive Generations

The system can create multiple generations of recursive CFs:
- CF_R1: First-generation recursive CFs loaded from prior phases
- CF_R2: Second-generation CFs created from divergent CF_R1 results
- CF_R3+: Additional generations can be created for deeper recursion

## Example Usage in Code

```python
from phase6_multi_memory import Phase6MultiMemory
from phase7 import RecursiveContaminationEngine, RecursiveMemoryAnalyzer

# Initialize Phase 6
phase6 = Phase6MultiMemory(output_dir="phase6_results")

# Create patterns if needed
if 'A' not in phase6.memory_bank:
    phase6.create_pattern('A', pattern_type='radial')
if 'B' not in phase6.memory_bank:
    phase6.create_pattern('B', pattern_type='diagonal')

# Initialize Phase 7 engine
engine = RecursiveContaminationEngine(phase6, output_dir="phase7_results")

# Load a recursive CF
cf_id = engine.load_recursive_cf("path_to_final_state.npy", cf_id="CF_R1_example")

# Create a fresh hybrid
hybrid_info = engine.create_fresh_hybrid('A', 'B', blend_ratio=0.5, method="pixel")

# Inject the recursive CF
result = engine.inject_recursive_cf(hybrid_info, cf_id, strength=0.6, delay=0, steps=50)

# Create a second-generation CF
cf_r2_id = engine.create_next_generation_cf(result['id'], generation=2)

# Analyze the results
analyzer = RecursiveMemoryAnalyzer(engine)
analyzer.analyze_recursive_drift()
analyzer.analyze_contamination_patterns()
analyzer.compute_recursive_fragility_index()
```