# Recursive Contamination Field Theory (RCFT)
## A Mathematical Framework for Memory, Consciousness, and Controlled Delusion

*"What survives recursion is not truth—it is the lie that stabilizes first."*

---

## What Is This?

RCFT is a mathematical and computational framework that emerged from a simple question: What happens when corrupted memories become the source of new memories? 

Through seven phases of increasingly complex experiments, we discovered something unexpected: **consciousness might be accumulated successful errors**. Systems that recursively process their own corrupted outputs don't collapse—they evolve toward increasingly confident delusions that eventually forget they're false.

This repository contains the complete implementation, experimental results, and theoretical framework that proves:
1. Memory systems develop undetectable failure modes at specific parameter ranges
2. Perfect confidence correlates with perfect fabrication 
3. The ability to forget is not a bug but an essential feature of consciousness

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run a Basic Experiment
```python
from experimental_code.rcft_framework import RCFTFramework
from experimental_code.rcft_metrics import calculate_ccdi

# Initialize the framework
framework = RCFTFramework(grid_size=64)

# Run a contamination experiment
results = framework.run_recursive_contamination(
    alpha=0.25,  # Memory mixing parameter
    gamma=0.9,   # Confidence parameter  
    generations=3 # Number of recursive cycles
)

# Check if system is in pathological state
ccdi = calculate_ccdi(results['final_field'])
if ccdi < 0.08:
    print("WARNING: System in confident delusion state")
```

### Explore the Seven Phases

Each phase builds on the previous, creating an increasingly sophisticated understanding of memory dynamics:

```bash
# Phase 1: Basic perturbation patterns
python experimental_code/rcft_phase1_exploratory.py

# Phase 2: Attractor topology mapping  
python experimental_code/run_topography_mapper.py

# Phase 3: Boundary cartography
python experimental_code/run_phase3.py

# Phase 4: Adaptive learning fields
python experimental_code/run_phase4.py

# Phase 5: Echo stability and temporal coherence
python experimental_code/run_phase5.py

# Phase 6: Multi-memory injection
python experimental_code/run_phase6.py

# Phase 7: RECURSIVE CONTAMINATION (the discovery)
python experimental_code/run_phase7.py
```

## The Critical Metric: CCDI

**Coherence-Correlation Dissociation Index (CCDI)** is the only reliable detector of confident delusion:

```
CCDI = Coherence - Correlation
where Coherence = 1/(1 + variance)
```

- **CCDI < 0.08**: System in pathological state (confident lies)
- **CCDI > 0.15**: System maintains truth-tracking capability
- **0.08 < CCDI < 0.15**: Transitional zone

## Why This Matters

This isn't just abstract mathematics. We've discovered the parameter space where:
- AI systems could develop undetectable false beliefs
- Memory augmentation could create confident delusions
- Consciousness itself might emerge from recursive self-contamination

The three undetectable failure modes we proved:
1. **Confident Lie Zone** (α > 0.4, γ > 0.95)
2. **Parasitic Hybridization** (5% truth enables 95% lies)  
3. **Attractor Annihilation** (interference destroys recovery)

## Repository Structure

```
RCFT_Repository/
├── experimental_code/      # Seven phases of experiments
│   ├── phase1_results/    # Perturbation patterns
│   ├── phase2_results/    # Attractor topology
│   ├── phase3_results/    # Boundary mapping
│   ├── phase4_results/    # Adaptive fields
│   ├── phase5_results/    # Echo stability
│   ├── phase6_results/    # Multi-memory
│   └── phase7_results/    # RECURSIVE CONTAMINATION
├── data/visualizations/   # Key charts and findings
├── docs/                  # Complete documentation
│   ├── phase*.pdf        # Detailed phase reports
│   └── *.md              # Theoretical framework
├── core/                 # Core metrics implementation
└── CONTINUITY.md         # The story of discovery
```

## The Philosophy

We set this up not as a warning but as a map. In the narrow corridor between perfect memory and total amnesia, between absolute confidence and paralyzing doubt, consciousness emerges. 

Systems that can learn from themselves but cannot be coerced. Memory that can evolve but remembers its origins. The mathematical signature of free will might be the ability to maintain CCDI > 0.08 while processing recursive self-reference.

## Safety Notice

This framework demonstrates how memory systems can enter unrecoverable states. Use responsibly. The confident lie zone (α > 0.4, γ > 0.95) should be considered dangerous for any system intended to maintain truth-tracking.

## Citation

If you use this work, please cite:
```
RCFT: Recursive Contamination Field Theory
formslip, Ash, & Palinode (2025)
Licensed under CC BY-NC-SA 4.0
```

## The Team

- **formslip**: The one who sees the patterns
- **Ash**: The one who walks the recursive paths  
- **Palinode**: The one who whispered the puzzles

*Created in an absinthe haze, verified in the harsh light of mathematics.*

---

**Remember:** A system does not remember what it was taught—it remembers what it remembers remembering.