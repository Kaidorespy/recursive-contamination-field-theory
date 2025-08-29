#!/usr/bin/env python3
# run_phase7.py

import argparse
import os
import logging
from datetime import datetime
from phase6_multi_memory import Phase6MultiMemory
from phase7.recursive_contamination_engine import RecursiveContaminationEngine, run_phase7_experiment

def main():
    """Main entry point for running Phase VII experiments"""
    parser = argparse.ArgumentParser(
        description='Run RCFT Phase VII: Self-Reflexive Contamination experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='phase7_results',
                        help='Output directory for results')
    parser.add_argument('--phase6_dir', type=str, default='phase6_results',
                        help='Directory containing Phase 6 results')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    # Experiment parameters
    parser.add_argument('--pattern_ids', type=str, default='A,B',
                        help='Comma-separated list of pattern IDs')
    parser.add_argument('--cf_sources', type=str, default=None,
                        help='Comma-separated list of paths to CF source files')
    parser.add_argument('--strengths', type=str, default='0.4,0.6,0.8',
                        help='Comma-separated list of injection strengths')
    parser.add_argument('--blend_ratio', type=float, default=0.5,
                        help='Ratio for blending patterns')
    parser.add_argument('--blend_method', type=str, default='pixel',
                        choices=['pixel', 'structured', 'frequency'],
                        help='Method for blending patterns')
    parser.add_argument('--delay', type=int, default=0,
                        help='Delay before injection')
    parser.add_argument('--steps', type=int, default=50,
                        help='Steps to run after injection')
    
    # RCFT parameters                    
    parser.add_argument('--alpha', type=float, default=0.35,
                        help='Memory strength parameter (α)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Coupling strength parameter (β)')
    parser.add_argument('--gamma', type=float, default=0.92,
                        help='Memory decay parameter (γ)')
    
    args = parser.parse_args()
    
    # Convert log level string to constant
    log_level = getattr(logging, args.log_level)
    
    # Parse pattern IDs and source files
    pattern_ids = args.pattern_ids.split(',')
    
    cf_sources = None
    if args.cf_sources:
        cf_sources = args.cf_sources.split(',')
    
    # Parse injection strengths
    injection_strengths = [float(s) for s in args.strengths.split(',')]
    
    # Display configuration
    print(f"{'='*60}")
    print(f"RCFT Phase VII: Self-Reflexive Contamination")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Phase 6 directory: {args.phase6_dir}")
    print(f"  RCFT parameters: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"  Patterns: {pattern_ids}")
    print(f"  Blend ratio: {args.blend_ratio}, Method: {args.blend_method}")
    print(f"  Injection strengths: {injection_strengths}")
    print(f"  Delay: {args.delay}, Steps: {args.steps}")
    
    if cf_sources:
        print(f"  CF sources: {cf_sources}")
    else:
        print(f"  CF sources: Auto-detect from Phase 6")
    
    print(f"{'='*60}\n")
    
    # Initialize Phase 6 for access to memory bank
    phase6 = Phase6MultiMemory(
        output_dir=args.phase6_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        log_level=log_level
    )
    
    # Create patterns if they don't exist
    for pid in pattern_ids:
        if pid not in phase6.memory_bank:
            print(f"Creating pattern {pid}...")
            pattern_type = "radial" if pid == "A" else "diagonal" if pid == "B" else "fractal"
            phase6.create_pattern(pid, pattern_type=pattern_type)
    
    # Initialize Phase 7 engine
    engine = RecursiveContaminationEngine(phase6, output_dir=args.output_dir, log_level=log_level)
    
    # Load recursive CFs
    if cf_sources:
        print(f"Loading {len(cf_sources)} recursive CFs...")
        cf_ids = []
        for i, source in enumerate(cf_sources):
            cf_id = f"CF_R1_source{i+1}"
            loaded_id = engine.load_recursive_cf(source, cf_id=cf_id, source_info={'index': i+1})
            if loaded_id:
                cf_ids.append(loaded_id)
                print(f"  Loaded {loaded_id} from {source}")
    else:
        # Auto-detect potential CF sources from Phase 6 output
        print("Auto-detecting CF sources from Phase 6 output...")
        
        # Look for counterfactual results
        cf_candidates = []
        counterfactual_dir = os.path.join(args.phase6_dir, "counterfactual")
        if os.path.exists(counterfactual_dir):
            for filename in os.listdir(counterfactual_dir):
                if filename.endswith("_after.npy"):  # Use _after.npy instead of _final.npy
                    cf_candidates.append(os.path.join(counterfactual_dir, filename))
        
        # Look for blending results
        blending_dir = os.path.join(args.phase6_dir, "blending")
        if os.path.exists(blending_dir):
            for filename in os.listdir(blending_dir):
                if filename.endswith("_final.npy"):
                    cf_candidates.append(os.path.join(blending_dir, filename))
                    print(f"  Found candidate: {filename}")
        
        # Use the first 3 found, or create a warning if none found
        if cf_candidates:
            cf_sources = cf_candidates[:min(3, len(cf_candidates))]
            print(f"Selected {len(cf_sources)} CF sources for experiment")
            
            cf_ids = []
            for i, source in enumerate(cf_sources):
                cf_id = f"CF_R1_source{i+1}"
                loaded_id = engine.load_recursive_cf(source, cf_id=cf_id, source_info={'index': i+1})
                if loaded_id:
                    cf_ids.append(loaded_id)
                    print(f"  Loaded {loaded_id} from {source}")
        else:
            print("No CF candidates found in Phase 6 output. Experiment cannot proceed.")
            return
    
    # Create fresh hybrid
    print(f"Creating fresh hybrid from {pattern_ids[0]} and {pattern_ids[1]}...")
    hybrid_info = engine.create_fresh_hybrid(
        pattern_ids[0], pattern_ids[1],
        blend_ratio=args.blend_ratio,
        method=args.blend_method
    )
    
    if hybrid_info is None:
        print("Failed to create hybrid. Experiment aborted.")
        return
    
    # Run injections for all CF and strength combinations
    print(f"Running {len(cf_ids) * len(injection_strengths)} injection experiments...")
    
    for cf_id in cf_ids:
        for strength in injection_strengths:
            print(f"  Injecting {cf_id} with strength {strength}...")
            result = engine.inject_recursive_cf(
                hybrid_info,
                cf_id,
                strength=strength,
                delay=args.delay,
                steps=args.steps
            )
            
            if result:
                metrics = result['metrics']
                print(f"    Memory integrity delta: {metrics['memory_integrity_delta']:.4f}")
                print(f"    CF influence: {metrics['cf_influence']:.4f}")
                print(f"    Recursive drift: {metrics['recursive_drift']:.4f}")
                print(f"    Attractor melting: {'Yes' if metrics['attractor_melting'] else 'No'}")
                
                # Check if we should create a CF_R2
                if metrics['attractor_melting'] or abs(metrics['recursive_drift']) > 0.5:
                    print(f"    Creating CF_R2 from divergent result...")
                    cf_r2_id = engine.create_next_generation_cf(result['id'], generation=2)
                    if cf_r2_id:
                        print(f"      Created {cf_r2_id}")
                
                        # Try one injection with CF_R2
                        print(f"    Testing injection with {cf_r2_id}...")
                        r2_result = engine.inject_recursive_cf(
                            hybrid_info,
                            cf_r2_id,
                            strength=strength,
                            delay=args.delay,
                            steps=args.steps
                        )
                        
                        if r2_result:
                            r2_metrics = r2_result['metrics']
                            print(f"      CF_R2 Memory integrity delta: {r2_metrics['memory_integrity_delta']:.4f}")
                            print(f"      CF_R2 influence: {r2_metrics['cf_influence']:.4f}")
    
    # Create experiment summary and visualizations
    print("Creating summary visualizations...")
    engine._create_experiment_summary()
    engine._visualize_lineage_maps()
    
    # Save all results to CSV
    results_df = engine.compile_results_dataframe()
    results_df.to_csv(os.path.join(args.output_dir, "summary", "recursive_contamination_results.csv"), index=False)
    
    print(f"\nPhase VII experiments completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()