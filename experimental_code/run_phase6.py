#!/usr/bin/env python3
# run_phase6.py
import pandas as pd
import numpy as np
import argparse
import os
import logging
import seaborn as sns
from datetime import datetime
from phase6.counterfactual_disruptor import CounterfactualDisruptor

from phase6_multi_memory import Phase6MultiMemory

def main():


    """Main entry point for running Phase VI experiments"""
    parser = argparse.ArgumentParser(
        description='Run RCFT Phase VI: Multi-Memory Coexistence experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='phase6_results',
                        help='Output directory for results')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    #parser.add_argument('--run_interference', action='store_true',
    #                    help='Run interference mapping experiment')
    #parser.add_argument('--interference_gap', type=int, default=20,
     #                   help='Delay between encodings')
    
    # RCFT parameters                    
    parser.add_argument('--alpha', type=float, default=0.35,
                        help='Memory strength parameter (α)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Coupling strength parameter (β)')
    parser.add_argument('--gamma', type=float, default=0.92,
                        help='Memory decay parameter (γ)')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps for evolution')
                        
    # Experiment selection
    parser.add_argument('--run_all', action='store_true',
                        help='Run all experiments')
    parser.add_argument('--run_interference', action='store_true',
                        help='Run interference mapping experiment')
    parser.add_argument('--run_recombination', action='store_true',
                        help='Run recombination thresholds experiment')
    parser.add_argument('--run_counterfactual', action='store_true',
                        help='Run counterfactual intrusion experiment')
    parser.add_argument('--run_switching', action='store_true',
                        help='Run cue-guided switching experiment')
    
    # Pattern configuration                    
    parser.add_argument('--pattern_ids', type=str, default='A,B,C',
                        help='Comma-separated list of pattern IDs')
    parser.add_argument('--blend_ratio', type=float, default=0.5,
                        help='Blending factor for recombination tests')
    parser.add_argument('--cue_strength', type=float, default=0.5,
                        help='Cue strength for context switching')
    parser.add_argument('--cf_strength', type=float, default=0.4,
                        help='Counterfactual injection similarity')
    parser.add_argument('--interference_gap', type=int, default=20,
                        help='Delay between encodings')
    
    parser.add_argument('--cf_strength_list', type=str, default='0.4',
                        help='Comma-separated list of counterfactual similarity strengths')
    parser.add_argument('--intrusion_levels', type=str, default='0.3,0.6,0.9',
                        help='Comma-separated list of intrusion strengths')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trials to run for each parameter combination')
    # Add to argument parser
    parser.add_argument('--run_hybrid_switching', action='store_true',
                        help='Run context switching after hybridization experiment')
    parser.add_argument('--echo_depth', type=int, default=0,
                  help='Depth of recursive echo buffer for memory protection (0 = none)')
    parser.add_argument('--cf_repetitions', type=int, default=1,
                  help='Number of sequential counterfactual exposures to apply')
    parser.add_argument('--cf_injection_delay', type=int, default=0,
                  help='Number of steps to wait after encoding before injecting counterfactual')
  
    parser.add_argument('--blend_method', type=str, default='pixel',
                        choices=['pixel', 'structured', 'frequency'],
                        help='Method for blending patterns')

   
    parser.add_argument('--run_recovery_ridge', action='store_true',
                    help='Run recovery ridge mapping experiment')
    parser.add_argument('--bias_type', type=str, default='none',
                        choices=['none', 'fingerprint', 'gradient', 'context'],
                        help='Type of bias to apply during recovery')
    parser.add_argument('--context_switch_strength', type=float, default=0.5,
                        help='Strength of context switch cue (alias for cue_strength)')
                        
    args = parser.parse_args()
    
 
    # Convert log level string to constant
    log_level = getattr(logging, args.log_level)
    
    # Parse pattern IDs
    pattern_ids = args.pattern_ids.split(',')
    
    # Display configuration
    print(f"{'='*60}")
    print(f"RCFT Phase VI: Multi-Memory Coexistence")
    print(f"{'='*60}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  RCFT parameters: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"  Patterns: {pattern_ids}")
    print(f"  Blend ratio: {args.blend_ratio}")
    print(f"  Cue strength: {args.cue_strength}")
    print(f"  Counterfactual strength: {args.cf_strength}")
    print(f"  Interference gap: {args.interference_gap}")
    print(f"{'='*60}\n")
    
    # Initialize Phase VI
    phase6 = Phase6MultiMemory(
        output_dir=args.output_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        max_steps=args.max_steps,
        log_level=log_level
    )
    
    # Create patterns
    for pid in pattern_ids:
        pattern_type = "fractal"  # Default
        if pid == "A":
            pattern_type = "radial"
        elif pid == "B":
            pattern_type = "diagonal"
            
        phase6.create_pattern(pid, pattern_type=pattern_type)
    
    # Run selected experiments
    results = {}
    
    if args.run_all:
        results = phase6.run_all_experiments()
    else:
        if args.run_interference:
            results['interference'] = phase6.run_interference_mapping(
                pattern_ids=pattern_ids[:2],
                delays=[args.interference_gap]
            )
            
        if args.run_recombination:
            results['recombination'] = phase6.run_recombination_thresholds(
                pattern_ids=pattern_ids[:2],  # Use first two patterns (A and B)
                blend_ratios=[args.blend_ratio],  # Test specified blend ratio
                perturbation_types=["flip"],  # Default perturbation type
                blend_methods=[args.blend_method]  # Use specified blend method
            )
            print(f"Recombination test complete. Results saved to {args.output_dir}")
            
        if args.run_counterfactual:
            # Parse the lists
            cf_strengths = [float(x) for x in args.cf_strength_list.split(',')]
            intrusion_strengths = [float(x) for x in args.intrusion_levels.split(',')]
            
            # Create a new disruptor with the specified echo depth
            from phase6.counterfactual_disruptor import CounterfactualDisruptor
            phase6.disruptor = CounterfactualDisruptor(phase6, echo_depth=args.echo_depth)
            
            results['counterfactual'] = phase6.disruptor.run_counterfactual_experiment(
                pattern_ids=pattern_ids[:2],  # Use first two patterns (A and B)
                counterfactual_similarity=cf_strengths,  # Use list of similarity values
                intrusion_strengths=intrusion_strengths,  # Use list of intrusion strengths
                trials=args.trials,  # Run multiple trials
                repetitions=args.cf_repetitions,  # Number of sequential exposures
                injection_delay=args.cf_injection_delay  # Delay before injection
            )
            print(f"Counterfactual intrusion test complete. Results saved to {args.output_dir}")                                                                              
            
        # In run_phase6.py - inside the main() function
        if args.run_switching:
            results['switching'] = phase6.run_cue_guided_switching(
                pattern_ids=pattern_ids[:2],  # Use first two patterns (A and B)
                cue_strengths=[args.cue_strength],  # Use provided cue_strength
                context_variations=["spatial", "gradient", "frequency"]  # Default context variations
            )
            print(f"Cue-guided switching test complete. Results saved to {args.output_dir}")
        # Add to the experiment execution section
        # In the hybrid_switching handler section
    # Replace the hybrid_switching section in run_phase6.py with this improved code:

        if args.run_hybrid_switching:
            # First create a blend
            blend_results = phase6.run_recombination_thresholds(
                pattern_ids=pattern_ids[:2],
                blend_ratios=[args.blend_ratio],
                perturbation_types=["flip"]
            )
            
            # Create a named hybrid pattern from the blend
            hybrid_id = f"{pattern_ids[0]}_{pattern_ids[1]}_blend"
            
            # Extract the hybrid pattern from the results and save it
            # Use proper DataFrame handling
            blend_row = blend_results[(blend_results['blend_ratio'] == args.blend_ratio) & 
                                    (blend_results['blend_method'] == 'pixel')]
            
            if not blend_row.empty:
                # Use the first matching row
                result_hybrid_id = blend_row.iloc[0]['hybrid_id']
                
                # Use the final state as the hybrid pattern
                hybrid_pattern = np.load(os.path.join(phase6.output_dir, "blending", 
                                                f"{result_hybrid_id}_final.npy"))
                
                # Create a memory trace for this hybrid using the same class as existing traces
                # Get the MemoryTrace class from the existing memory bank
                MemoryTrace = type(phase6.memory_bank[pattern_ids[0]])
                
                # Create a new memory trace for the hybrid and add it to the memory bank
                phase6.memory_bank[hybrid_id] = MemoryTrace(hybrid_id, hybrid_pattern)
                
                # Now run switching from hybrid to pure attractors
                results['hybrid_switching'] = phase6.run_cue_guided_switching(
                    pattern_ids=[hybrid_id, pattern_ids[0], pattern_ids[1]],
                    cue_strengths=[args.cue_strength],
                    context_variations=["spatial", "gradient"]
                )
                
                # Create a comparison report
                print(f"Hybrid switching test complete. Results saved to {args.output_dir}")
            else:
                print(f"No matching blend found with ratio {args.blend_ratio} and pixel method")
        if args.run_recovery_ridge:
            # If context_switch_strength is provided, use it for cue_strength
            if args.context_switch_strength != 0.5:  # If it was explicitly set
                args.cue_strength = args.context_switch_strength
            
            # Create the recovery ridge mapping experiment object
            from phase6.recovery_ridge_mapper import RecoveryRidgeMapper
            mapper = RecoveryRidgeMapper(phase6)
            
            # Run the recovery ridge mapping experiment
            results['recovery_ridge'] = mapper.run_recovery_ridge_experiment(
                pattern_ids=pattern_ids[:2],  # Use first two patterns (A and B)
                blend_method=args.blend_method,
                bias_type=args.bias_type,
                cue_strength=args.cue_strength,
                blend_ratio=args.blend_ratio
            )
            print(f"Recovery ridge mapping complete. Results saved to {args.output_dir}")
    
    print(f"\nPhase VI experiments completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()