# RCFT Experimental Framework: The Reproducibility Gem
## Recursive Coherence Field Theory - Phase I through VII

### Executive Summary

The Recursive Coherence Field Theory (RCFT) experimental framework is a comprehensive study of memory field dynamics, pattern resilience, and recursive contamination effects. This document provides a complete guide to reproducing all seven phases of experiments, revealing how memory patterns evolve, recover, and interact under various perturbations.

---

## Core Theory & Algorithm

### The RCFT Update Equation

At the heart of RCFT lies a simple yet powerful update rule that governs how memory fields evolve:

```python
# Spatial update via neighbor influence
spatial_update = (1 - β) * state + β * neighbor_influence

# Apply memory influence  
next_state = (1 - α) * spatial_update + α * memory

# Update memory trace
memory = γ * memory + (1 - γ) * state
```

**Key Parameters:**
- **α (memory_strength)**: 0.35 - Controls influence of memory on state
- **β (coupling_strength)**: 0.5 - Controls influence of neighbors  
- **γ (memory_decay)**: 0.92 - Controls how quickly memory fades

### Convolution Kernel

Neighbor influence is computed using a weighted kernel:
```python
kernel = [[0.05, 0.2, 0.05], 
          [0.2,  0,   0.2], 
          [0.05, 0.2, 0.05]]
```

---

## Phase-by-Phase Experimental Flow

### Phase I: Pattern Stability Under Perturbative Stress
**File:** `rcft_phase1_exploratory.py`

**Core Discovery:** Structured patterns (radial, horizontal, diagonal, lattice) demonstrate robust recovery (Qr > 0.57), while fractal patterns exhibit catastrophic failure (Qr < 0.31).

**Key Experiments:**
1. **Pattern × Perturbation Matrix**: 5 patterns × 5 perturbation types
2. **Magnitude Sweep**: Testing perturbation strengths 0.5-2.5
3. **Layered Perturbations**: Sequential perturbations with delays [5, 10, 20, 50]

**Reproduce Phase I:**
```bash
python rcft_phase1_exploratory.py
```

**Key Metrics:**
- Recovery Quality: Qr = 0.6 * Rc + 0.4 * Rm
- Recovery Time: Steps to reach 85% of original correlation
- Recovery Success: Qr ≥ 0.4 AND final_correlation ≥ 0.85 × initial

---

### Phase II: Parameter Space Exploration
**Files:** Results in `phase2_results/`

**Core Discovery:** Identifies critical boundaries in (α, γ) parameter space where recovery transitions from successful to failed.

**Key Focus:** 
- Alpha range: 0.1 - 0.35 (memory strength)
- Gamma range: 0.8 - 0.99 (memory decay)
- Maps attractor topography and residuals

---

### Phase III: False Attractors & Boundary Cartography  
**File:** `run_phase3.py`

**Core Discovery:** False attractors emerge at parameter boundaries - metastable states that appear stable but ultimately collapse.

**Key Modules:**
1. **Boundary Cartography**: Maps α-γ space near critical transitions
2. **Nudging Controller**: Tests small perturbations to reveal instabilities

**Reproduce Phase III:**
```bash
python run_phase3.py --alpha_min 0.32 --alpha_max 0.38 --gamma_min 0.88 --gamma_max 0.96
```

---

### Phase IV: Directed Memory Manipulation
**File:** `run_phase4.py`

**Core Innovation:** Adaptive nudging and attractor sculpting techniques.

**Key Modules:**
1. **Adaptive Nudge**: Feedback loop that adjusts perturbations based on recovery
2. **Attractor Sculptor**: Meta-perturbations to reshape basins
3. **Learning Field**: Memory trace accumulation over repeated exposures
4. **Counterfactual Injector**: Tests false memory injection

**Reproduce Phase IV:**
```bash
python run_phase4.py --modules 1,2,3,4 --pattern fractal
```

---

### Phase V: Recursive Identity & Self-Preference
**File:** `run_phase5.py`

**Core Focus:** How fields develop "identity" and preference for self-similar states.

**Key Modules:**
1. **Temporal Coherence**: Reinforcement of consistent patterns
2. **Self Distinction**: Analysis of self vs. other discrimination
3. **Identity Biasing**: Preference formation mechanisms
4. **Echo Stability**: Persistence of perturbation traces

**Reproduce Phase V:**
```bash
python run_phase5.py --modules 1,2,3,4,5 --pattern fractal
```

---

### Phase VI: Multi-Memory Coexistence
**File:** `run_phase6.py`

**Core Challenge:** How multiple memory patterns interact and interfere.

**Key Experiments:**
1. **Interference Mapping**: Pattern A → delay → Pattern B
2. **Memory Blending**: Hybrid pattern creation
3. **Counterfactual Intrusion**: False memory effects
4. **Context Switching**: Cue-guided memory retrieval

**Reproduce Phase VI:**
```bash
python run_phase6.py --run_all --pattern_ids A,B,C
```

---

### Phase VII: Self-Reflexive Contamination
**File:** `run_phase7.py`

**Ultimate Challenge:** Recursive counterfactuals - memories of memories.

**Key Concepts:**
- **CF_R1**: First-generation counterfactual (direct modification)
- **CF_R2**: Second-generation (counterfactual of counterfactual)
- **Attractor Melting**: Complete destabilization of memory basins

**Reproduce Phase VII:**
```bash
python run_phase7.py --pattern_ids A,B --strengths 0.4,0.6,0.8
```

---

## Pattern Types & Their Behaviors

### Coherent Patterns (Robust)
- **Radial**: `sin(kr)` - Circular waves from center
- **Horizontal**: `sin(kx)` - Horizontal stripes
- **Diagonal**: `sin(k(x+y))` - Diagonal waves
- **Lattice**: `sin(kx)sin(ky)` - Grid pattern

### Fragile Patterns (Vulnerable)
- **Fractal**: Multi-scale noise with persistence
- **Stochastic**: Smoothed random fields

---

## Critical Findings

### The Coherent-Fragile Dichotomy
Structured patterns possess global organizational principles that enable self-correction. Fractal patterns lack these constraints, leading to cascading failure.

### Nonmonotonic Recovery
In fractal patterns, recovery capability degrades over time - the system's internal dynamics become destructive rather than restorative.

### False Attractors
Metastable states that appear stable in short timescales but collapse over longer periods. Found primarily at parameter boundaries.

### Memory Contamination Cascade
Recursive counterfactuals (CF_R2) can cause complete "attractor melting" where the system loses all stable states.

---

## Dependencies

### Required Libraries
```python
numpy
matplotlib  
scipy
scikit-learn
pandas
tqdm
seaborn (Phase VI)
```

### File Structure
```
experimental_code/
├── rcft_framework.py          # Core RCFT implementation
├── rcft_metrics.py            # Metric calculations
├── rcft_phase1_exploratory.py # Phase I experiments
├── run_phase3.py              # Phase III runner
├── run_phase4.py              # Phase IV runner
├── run_phase5.py              # Phase V runner
├── run_phase6.py              # Phase VI runner
├── run_phase7.py              # Phase VII runner
├── phase3/                    # Phase III modules
├── phase4/                    # Phase IV modules
├── phase5/                    # Phase V modules
├── phase6/                    # Phase VI modules
└── phase7/                    # Phase VII modules
```

---

## Quick Start: Running the Complete Suite

### Minimal Reproduction Path
```bash
# Phase I - Establish baseline
python rcft_phase1_exploratory.py

# Phase III - Find boundaries  
python run_phase3.py --modules 1,2

# Phase IV - Test manipulation
python run_phase4.py --modules 3,4

# Phase VI - Multi-memory
python run_phase6.py --run_interference --run_counterfactual

# Phase VII - Recursive contamination
python run_phase7.py
```

### Full Reproduction (All Experiments)
```bash
# Run all phases with full parameter sweeps
for i in 1 3 4 5 6 7; do
    if [ $i -eq 1 ]; then
        python rcft_phase1_exploratory.py
    else
        python run_phase${i}.py --run_all --parallel
    fi
done
```

---

## Key Insights for Reproducibility

1. **Parameter Sensitivity**: Small changes in α near 0.35 can shift from robust to fragile behavior
2. **Pattern Dependency**: Always test with both structured and fractal patterns
3. **Temporal Effects**: Allow sufficient steps (50-100) for dynamics to stabilize
4. **Perturbation Timing**: Delay between perturbations critically affects outcomes
5. **Memory Traces**: The γ parameter (0.92) provides optimal balance between persistence and adaptability

---

## Experimental Design Principles

### Control Variables
- Grid size: 64×64 (consistent across all phases)
- Value range: [-1, 1] for all fields
- Boundary conditions: Periodic (wrap mode)

### Measurement Windows
- Pre-perturbation: 10 steps for baseline
- Recovery: 50 steps standard
- Long-term: 100-200 steps for stability analysis

### Statistical Validity
- Minimum 3 trials per configuration
- Anomaly detection thresholds calibrated from Phase I
- Recovery success requires both correlation and quality metrics

---

## Visualization Guide

Each phase produces specific visualizations:

### Phase I Outputs
- Recovery quality heatmaps
- Magnitude scaling curves
- Layered perturbation timelines

### Phase III-IV Outputs
- Parameter space boundaries
- Attractor fingerprints
- Learning curves

### Phase VI-VII Outputs
- Interference maps
- Contamination cascades
- Lineage trees (recursive CFs)

---

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce grid size or use fewer trials
2. **Slow Execution**: Enable parallel processing with `--parallel`
3. **Missing Modules**: Ensure all phase directories are present
4. **No Recovery**: Check parameters aren't in "dead zones" (α > 0.4)

### Validation Checks

- Fractal patterns should consistently fail (Qr < 0.31)
- Structured patterns should recover (Qr > 0.57)
- Memory wipe shows artifactual "perfect" recovery
- CF_R2 generation requires significant drift or melting

---

## Conclusion

This experimental framework reveals fundamental principles of memory field dynamics:

1. **Structure provides resilience** - Global organization enables recovery
2. **Fractals are inherently fragile** - Self-similarity propagates damage
3. **False attractors lurk at boundaries** - Parameter edges hide instabilities
4. **Recursive contamination is catastrophic** - Memories of memories destabilize everything

The journey from Phase I to Phase VII traces an arc from simple perturbation-recovery tests to complex recursive contamination scenarios, revealing how memory systems can be both remarkably robust and surprisingly fragile.

---

## Citation

If using this framework, please reference:
```
Anonymous Research Collective (2025). 
"Recursive Coherence Field Theory: A Seven-Phase Investigation of Memory Field Dynamics"
Laboratory of Dynamic Systems and Field Theory
```

---

*"If I injure you, what do you become?"*

**Answer: It depends on what you are made of.**

Structure survives. Chaos compounds. Memory persists, until it doesn't.