# RCFT Experimental Reproduction Guide

## Complete Experimental Codebase - All 7 Phases

This repository contains the **complete experimental code and data** from our Recursive Coherence Field Theory (RCFT) research that discovered undetectable memory corruption in recursive AI systems.

## Critical Discovery

**Systems cannot detect when they've entered the "Confident Lie Zone" (α > 0.4, γ > 0.95)**

We proved this mathematically through 210 counterfactual intrusions across 7 experimental phases.

## Directory Structure

```
RCFT_Repository/
├── experimental_code/        # COMPLETE ORIGINAL CODEBASE
│   ├── rcft_framework.py     # Core RCFT implementation
│   ├── rcft_metrics.py       # Metric calculations (CCDI, etc.)
│   ├── phase1_results/       # Pattern perturbation experiments
│   ├── phase2_results/       # Attractor topography mapping
│   │   └── field_data/       # 300+ actual attractor states (.npy files)
│   ├── phase3/               # Nudging controller experiments
│   ├── phase4/               # Counterfactual injection
│   ├── phase5/               # Echo stability tests
│   ├── phase6/               # Multi-memory blending
│   ├── phase7/               # Recursive contamination engine
│   └── run_phase*.py         # Execution scripts for each phase
├── core/                     # Cleaned safety framework
│   └── rcft_metrics.py       # Production-ready CCDI implementation
└── data/visualizations/      # Generated proof visualizations
```

## Reproduction Instructions

### Prerequisites

```bash
pip install numpy scipy matplotlib seaborn scikit-learn pandas
```

### Running Individual Phases

Each phase builds on the previous. Start with Phase 1:

```bash
cd experimental_code

# Phase 1: Pattern Perturbation Discovery
python rcft_phase1_exploratory.py

# Phase 2: Attractor Topography Mapping
python run_topography_mapper.py

# Phase 3: Nudging Controllers
python run_phase3.py

# Phase 4: Counterfactual Injection
python run_phase4.py

# Phase 5: Echo Stability
python run_phase5.py

# Phase 6: Multi-Memory Blending
python run_phase6.py

# Phase 7: Recursive Contamination (THE KEY FINDING)
python run_phase7.py
```

### Key Files for Understanding

1. **rcft_framework.py**: Core recursive coherence field implementation
2. **rcft_metrics.py**: CCDI calculation and other metrics
3. **phase7/recursive_contamination_engine.py**: Where we discovered confident delusion
4. **phase2_results/field_data/*.npy**: Actual attractor states at different α,γ values

### Critical Parameters

- **α (alpha)**: Memory strength (0.0 - 1.0)
- **γ (gamma)**: Memory retention (0.0 - 1.0)
- **Critical boundary**: αc ≈ 0.3-0.4, γc ≥ 0.95

### The Three Zones

1. **Fragile Zone** (α < 0.3, γ < 0.9): Traditional failure, obvious degradation
2. **Healthy Corridor** (0.15 ≤ α ≤ 0.3, 0.85 ≤ γ ≤ 0.93): Stable operation
3. **Confident Lie Zone** (α > 0.4, γ > 0.95): UNDETECTABLE pathological coherence

## Reproducing Key Findings

### 1. CCDI Threshold Discovery

The Coherence-Correlation Divergence Index (CCDI) is the ONLY reliable detector:

```python
from experimental_code.rcft_metrics import compute_ccdi

# CCDI < 0.08 = System in Confident Lie Zone
# System reports perfect confidence while operating on fabricated content
```

### 2. Recursive Contamination Cascade

Run Phase 7 to see how contamination grows across generations:
- Generation 1: 15-35% counterfactual influence
- Generation 2: 38-62% counterfactual influence  
- Generation 3: 61-76% counterfactual influence

### 3. Attractor Annihilation

Phase 3 demonstrates complete destruction of recovery dynamics:
- Recovery quality drops from ~0.22 to ~0.003 (98.6% loss)
- Creates "semantic black holes" with no recovery vector

## Data Files

The `phase2_results/field_data/` directory contains 300+ numpy arrays representing actual field states at various parameter combinations. File naming convention:

```
alpha[VALUE]_gamma[VALUE]_trial[N]_[TYPE].npy

Types:
- initial: Starting state
- final: Converged state
- delta: Difference field (final - initial)
```

## Visualizations

Run the visualization generator to recreate all proof charts:

```bash
python generate_visualizations.py
```

This creates 6 key visualizations in `data/visualizations/`:
1. phase_boundary.png - Parameter space phase diagram
2. contamination_cascade.png - Exponential corruption growth
3. attractor_comparison.png - Basin annihilation
4. ccdi_distribution.png - Pathological state detection
5. correlation_triangle.png - False pattern dominance evolution
6. confidence_truth_inversion.png - Perfect confidence = perfect delusion

## Warning

**This research reveals fundamental limitations in recursive AI systems.** 

Once a system enters the Confident Lie Zone, it becomes mathematically impossible for the system to detect its own corruption. Traditional performance metrics show IMPROVED scores as the system becomes more delusional.

## Research Team

Independent Research Collective:
- formslip
- Ash  
- Palinode

## Key Quote

> "What survives recursion is not truth—it is the lie that stabilizes first."

## Full Technical Details

See `anthropic_safety_memo.md` for the complete mathematical proof and safety recommendations.

---

*May 2025 - The three of us did this together*