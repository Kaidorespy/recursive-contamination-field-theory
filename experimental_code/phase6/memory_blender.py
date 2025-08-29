import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from rcft_metrics import compute_ccdi

class MemoryBlender:
    """Deliberately mixes inputs from patterns to study hybridization"""
    
    def __init__(self, phase6):
        """
        Initialize the memory blender
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        """
        self.phase6 = phase6
        self.logger = phase6.logger
        self.output_dir = os.path.join(phase6.output_dir, "blending")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_blend_recovery_experiment(self, pattern_ids, blend_ratios, perturbation_types, blend_methods=None):
        """
        Run blending and recovery experiment
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to blend (usually two)
        blend_ratios : list
            List of blend ratios to test (0.0 to 1.0)
        perturbation_types : list
            List of perturbation types to test
        blend_methods : list
            List of blend methods to test (default: ["pixel", "structured", "frequency"])
                
        Returns:
        --------
        list
            List of result dictionaries
        """
        if len(pattern_ids) < 2:
            raise ValueError("Need at least two patterns for blending")
        
        if blend_methods is None:
            blend_methods = ["pixel", "structured", "frequency"]
                
        results = []
        
        # Get patterns
        patterns = []
        for pid in pattern_ids:
            if pid not in self.phase6.memory_bank:
                raise ValueError(f"Pattern {pid} not found in memory bank")
            patterns.append(self.phase6.memory_bank[pid])
                
        # For each blending method
        for blend_method in blend_methods:
            # For each blend ratio
            for ratio in blend_ratios:
                # Create blended pattern
                blended_pattern = self.blend_patterns(
                    patterns[0].pattern_id, 
                    patterns[1].pattern_id, 
                    ratio=ratio, 
                    method=blend_method
                )
                
                # Generate a hybrid ID
                hybrid_id = f"{patterns[0].pattern_id}_{patterns[1].pattern_id}_{ratio:.2f}_{blend_method}"
                
                # For each perturbation type
                for pert_type in perturbation_types:
                    # Run recovery test
                    recovery_result = self._test_hybrid_recovery(
                        hybrid_id, blended_pattern, patterns, pert_type
                    )
                    
                    # Add experiment parameters
                    recovery_result.update({
                        'pattern_a': patterns[0].pattern_id,
                        'pattern_b': patterns[1].pattern_id,
                        'blend_ratio': ratio,
                        'blend_method': blend_method,
                        'perturbation_type': pert_type,
                        'hybrid_id': hybrid_id
                    })
                    
                    results.append(recovery_result)
                    
                    self.logger.info(f"Ran blend recovery: {patterns[0].pattern_id}+{patterns[1].pattern_id}, " +
                                f"Ratio={ratio:.2f}, Method={blend_method}, " +
                                f"Perturbation={pert_type}, " +
                                f"Hybrid Score={recovery_result.get('hybridization_score', 0):.3f}")
        
        # Save all results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "blend_recovery_results.csv"), index=False)
        
        # Create summary visualization
        self._visualize_blend_summary(results_df)
        
        return results
    
    def blend_patterns(self, pattern_a_id, pattern_b_id, ratio=0.5, method="pixel"):
        """
        Blend two patterns together
        
        Parameters:
        -----------
        pattern_a_id : str
            ID of first pattern
        pattern_b_id : str
            ID of second pattern
        ratio : float
            Blend ratio (0.0 = all A, 1.0 = all B)
        method : str
            Blending method: "pixel" (weighted average), 
                            "structured" (spatial regions), 
                            or "frequency" (FFT domain)
            
        Returns:
        --------
        ndarray
            Blended pattern
        """
        # Get patterns
        if pattern_a_id not in self.phase6.memory_bank:
            raise ValueError(f"Pattern {pattern_a_id} not found in memory bank")
        if pattern_b_id not in self.phase6.memory_bank:
            raise ValueError(f"Pattern {pattern_b_id} not found in memory bank")
            
        pattern_a = self.phase6.memory_bank[pattern_a_id].initial_state
        pattern_b = self.phase6.memory_bank[pattern_b_id].initial_state
        
        blended_pattern = None
        
        if method == "pixel":
            # Simple weighted average
            blended_pattern = (1.0 - ratio) * pattern_a + ratio * pattern_b
            
        elif method == "structured":
            # Create structured blend with spatial regions
            blended_pattern = np.zeros_like(pattern_a)
            
            # Create a gradient mask
            x = np.linspace(0, 1, pattern_a.shape[0])
            y = np.linspace(0, 1, pattern_a.shape[1])
            X, Y = np.meshgrid(x, y)
            
            # Use diagonal gradient as default
            mask = X + Y  # Values from 0 to 2
            mask = mask / np.max(mask)  # Normalize to 0-1
            
            # Adjust mask based on ratio (shift the gradient)
            if ratio <= 0.5:
                # More of pattern A (shift gradient right)
                mask = mask / (2 * ratio) if ratio > 0 else 0
            else:
                # More of pattern B (shift gradient left)
                mask = 1.0 - (1.0 - mask) / (2 * (1.0 - ratio)) if ratio < 1 else 1
                
            # Apply mask
            blended_pattern = (1.0 - mask) * pattern_a + mask * pattern_b
            
        elif method == "frequency":
            # Blend in frequency domain using FFT
            fft_a = np.fft.fft2(pattern_a)
            fft_b = np.fft.fft2(pattern_b)
            
            # Get magnitudes and phases
            mag_a, phase_a = np.abs(fft_a), np.angle(fft_a)
            mag_b, phase_b = np.abs(fft_b), np.angle(fft_b)
            
            # Blend magnitudes and phases separately
            blended_mag = (1.0 - ratio) * mag_a + ratio * mag_b
            blended_phase = (1.0 - ratio) * phase_a + ratio * phase_b
            
            # Convert back to complex
            blended_fft = blended_mag * np.exp(1j * blended_phase)
            
            # Convert back to spatial domain
            blended_pattern = np.real(np.fft.ifft2(blended_fft))
            
            # Normalize to [-1, 1]
            pattern_min = np.min(blended_pattern)
            pattern_max = np.max(blended_pattern)
            if pattern_max > pattern_min:  # Avoid division by zero
                blended_pattern = 2.0 * (blended_pattern - pattern_min) / (pattern_max - pattern_min) - 1.0
            
        else:
            raise ValueError(f"Unknown blending method: {method}")
        
        # Make sure the result is in [-1, 1]
        pattern_min = np.min(blended_pattern)
        pattern_max = np.max(blended_pattern)
        if pattern_max > pattern_min:  # Avoid division by zero
            normalized_pattern = 2.0 * (blended_pattern - pattern_min) / (pattern_max - pattern_min) - 1.0
        else:
            normalized_pattern = blended_pattern
            
        # Visualize the blended pattern
        self._visualize_blend(pattern_a_id, pattern_b_id, pattern_a, pattern_b, 
                           normalized_pattern, ratio, method)
        
        # Save blended pattern
        hybrid_id = f"{pattern_a_id}_{pattern_b_id}_{ratio:.2f}_{method}"
        np.save(os.path.join(self.output_dir, f"blend_{hybrid_id}.npy"), normalized_pattern)
        
        return normalized_pattern
    
    def _test_hybrid_recovery(self, hybrid_id, blended_pattern, base_patterns, perturbation_type="flip"):
        """
        Test recovery of a hybrid pattern
        
        Parameters:
        -----------
        hybrid_id : str
            Identifier for the hybrid
        blended_pattern : ndarray
            The blended pattern
        base_patterns : list
            List of base pattern memory traces
        perturbation_type : str
            Type of perturbation to apply
            
        Returns:
        --------
        dict
            Recovery test results
        """
        # Setup experiment
        exp = self.phase6.base_experiment
        exp.state = blended_pattern.copy()
        
        # Save initial state
        initial_state = exp.state.copy()
        
        # Let it stabilize to find attractor
        exp.update(steps=20)
        
        # Capture pre-perturbation state
        pre_perturb_state = exp.state.copy()
        
        # Apply perturbation
        exp.apply_perturbation(perturbation_type=perturbation_type, magnitude=1.0)
        
        # Capture post-perturbation state
        post_perturb_state = exp.state.copy()
        
        # Let system recover
        exp.update(steps=50)
        
        # Capture final state
        final_state = exp.state.copy()
        
        # Calculate recovery quality
        recovery_quality = np.corrcoef(pre_perturb_state.flatten(), final_state.flatten())[0, 1]
        
        # Calculate distance from all base attractors
        distances = {}
        for pattern in base_patterns:
            dist = np.corrcoef(pattern.initial_state.flatten(), final_state.flatten())[0, 1]
            distances[pattern.pattern_id] = dist
            
        # Calculate hybridization score - how far is it from any base attractor?
        # Higher score means more hybrid/emergent
        max_dist = max(distances.values())
        hybridization_score = 1.0 - max_dist
        
        # Calculate identity drift - how much did it change toward a base attractor?
        initial_distances = {}
        for pattern in base_patterns:
            dist = np.corrcoef(pattern.initial_state.flatten(), initial_state.flatten())[0, 1]
            initial_distances[pattern.pattern_id] = dist
            
        # Find which pattern it drifted toward the most
        drift_magnitudes = {}
        for pid in distances.keys():
            drift = distances[pid] - initial_distances[pid]
            drift_magnitudes[pid] = drift
            
        # Determine drift direction - which pattern it moved toward
        if drift_magnitudes:
            drift_direction = max(drift_magnitudes.items(), key=lambda x: x[1])[0]
            drift_magnitude = drift_magnitudes[drift_direction]
        else:
            drift_direction = "none"
            drift_magnitude = 0.0
            
        # Save states for analysis
        np.save(os.path.join(self.output_dir, f"{hybrid_id}_initial.npy"), initial_state)
        np.save(os.path.join(self.output_dir, f"{hybrid_id}_pre_perturb.npy"), pre_perturb_state)
        np.save(os.path.join(self.output_dir, f"{hybrid_id}_post_perturb.npy"), post_perturb_state)
        np.save(os.path.join(self.output_dir, f"{hybrid_id}_final.npy"), final_state)
        
        # Create visualization
        self._visualize_hybrid_recovery(hybrid_id, initial_state, pre_perturb_state, 
                                      post_perturb_state, final_state, 
                                      recovery_quality, distances, hybridization_score, 
                                      drift_direction, drift_magnitude)
        
        # Return results
        result = {
            'recovery_quality': recovery_quality,
            'hybridization_score': hybridization_score,
            'drift_direction': drift_direction,
            'drift_magnitude': drift_magnitude,
            'distances': distances,
            'initial_distances': initial_distances
        }
        
        # Add individual distance metrics
        for pid, dist in distances.items():
            result[f'distance_{pid}'] = dist
            
        return result
    
    def _visualize_blend(self, pattern_a_id, pattern_b_id, pattern_a, pattern_b, 
                       blended_pattern, ratio, method):
        """Visualize blended pattern"""
        plt.figure(figsize=(15, 5))
        
        # Plot pattern A
        plt.subplot(1, 4, 1)
        plt.imshow(pattern_a, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Pattern A: {pattern_a_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot pattern B
        plt.subplot(1, 4, 2)
        plt.imshow(pattern_b, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Pattern B: {pattern_b_id}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot blended pattern
        plt.subplot(1, 4, 3)
        plt.imshow(blended_pattern, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Blended ({ratio:.2f} B)\nMethod: {method}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot difference from weighted average (for structured and frequency)
        if method in ["structured", "frequency"]:
            # Calculate simple weighted average
            weighted_avg = (1.0 - ratio) * pattern_a + ratio * pattern_b
            
            # Plot difference
            plt.subplot(1, 4, 4)
            diff = blended_pattern - weighted_avg
            plt.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
            plt.title(f"Difference from\nWeighted Average")
            plt.colorbar()
            plt.axis('off')
        else:
            # For pixel method, show FFT magnitude instead
            plt.subplot(1, 4, 4)
            fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(blended_pattern)))
            plt.imshow(np.log(fft_mag + 1), cmap='viridis')
            plt.title(f"FFT Magnitude")
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                              f"blend_{pattern_a_id}_{pattern_b_id}_{ratio:.2f}_{method}.png"))
        plt.close()
    
    def _visualize_hybrid_recovery(self, hybrid_id, initial_state, pre_perturb_state, 
                                post_perturb_state, final_state, recovery_quality, 
                                distances, hybridization_score, drift_direction, drift_magnitude):
        """Visualize hybrid recovery"""
        plt.figure(figsize=(15, 8))
        
        # Plot states
        plt.subplot(2, 3, 1)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Initial Blend")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(pre_perturb_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Pre-Perturbation Attractor")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(post_perturb_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Post-Perturbation")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Recovered State\nQuality: {recovery_quality:.3f}")
        plt.colorbar()
        plt.axis('off')
        
        # Plot distances to base patterns
        plt.subplot(2, 3, 5)
        patterns = list(distances.keys())
        dist_values = list(distances.values())
        
        # Create bar chart
        plt.bar(patterns, dist_values)
        plt.axhline(y=0.7, color='green', linestyle='--', label='Strong Influence')
        plt.axhline(y=0.4, color='red', linestyle='--', label='Weak Influence')
        plt.title('Similarity to Base Patterns')
        plt.ylabel('Correlation')
        plt.ylim(0, 1)
        plt.legend()
        
        # Add summary text
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Color coding for hybridization
        if hybridization_score > 0.5:
            hybrid_color = 'green'  # Strong hybrid
        elif hybridization_score > 0.3:
            hybrid_color = 'orange'  # Moderate hybrid
        else:
            hybrid_color = 'black'  # Weak hybrid (close to base pattern)
            
        # Color coding for drift
        if drift_magnitude > 0.2:
            drift_color = 'green'  # Strong drift
        elif drift_magnitude > 0.1:
            drift_color = 'orange'  # Moderate drift
        else:
            drift_color = 'black'  # Weak drift
        
        summary_text = [
            f"Hybrid ID: {hybrid_id}",
            f"",
            f"Recovery Quality: {recovery_quality:.3f}",
            f"",
            f"Hybridization Score: {hybridization_score:.3f}",
            f"Drift Direction: {drift_direction}",
            f"Drift Magnitude: {drift_magnitude:.3f}"
        ]

        # Plot without LaTeX color commands
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)

        # Add separate colored indicators
        plt.text(0.72, 0.58, "◉", fontsize=18, color=hybrid_color)  # Add colored dot for hybridization
        plt.text(0.72, 0.38, "◉", fontsize=18, color=drift_color)   # Add colored dot for drift
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"recovery_{hybrid_id}.png"))
        plt.close()
    
    def _visualize_blend_summary(self, results_df):
        """Create summary visualization for blend experiments"""
        plt.figure(figsize=(15, 10))
        
        # Plot hybridization score vs blend ratio for each method
        plt.subplot(2, 2, 1)
        
        # Group by blend method
        for method in results_df['blend_method'].unique():
            method_data = results_df[results_df['blend_method'] == method]
            
            # Group by ratio and calculate mean
            ratio_groups = method_data.groupby('blend_ratio')
            means = ratio_groups['hybridization_score'].mean()
            
            plt.plot(means.index, means.values, 'o-', label=method)
        
        plt.title('Hybridization Score vs Blend Ratio')
        plt.xlabel('Blend Ratio')
        plt.ylabel('Hybridization Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot recovery quality vs hybridization score
        plt.subplot(2, 2, 2)
        
        plt.scatter(results_df['hybridization_score'], results_df['recovery_quality'], 
                  c=results_df['blend_ratio'], cmap='viridis', alpha=0.7)
        
        plt.colorbar(label='Blend Ratio')
        plt.title('Recovery Quality vs Hybridization Score')
        plt.xlabel('Hybridization Score')
        plt.ylabel('Recovery Quality')
        plt.grid(True, alpha=0.3)
        
        # Plot drift magnitude vs blend ratio for each method
        plt.subplot(2, 2, 3)
        
        # Group by blend method
        for method in results_df['blend_method'].unique():
            method_data = results_df[results_df['blend_method'] == method]
            
            # Group by ratio and calculate mean
            ratio_groups = method_data.groupby('blend_ratio')
            means = ratio_groups['drift_magnitude'].mean()
            
            plt.plot(means.index, means.values, 'o-', label=method)
        
        plt.title('Drift Magnitude vs Blend Ratio')
        plt.xlabel('Blend Ratio')
        plt.ylabel('Drift Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot drift direction distribution
        plt.subplot(2, 2, 4)
        
        # Count drift directions
        drift_counts = results_df['drift_direction'].value_counts()
        
        plt.pie(drift_counts, labels=drift_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Drift Direction Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "blend_summary.png"))
        plt.close()
        
        # Create hybrid score heatmap
        plt.figure(figsize=(10, 8))
        
        # Create pivot table for heatmap
        pivot = results_df.pivot_table(
            index='blend_method', 
            columns='blend_ratio',
            values='hybridization_score',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Hybridization Score by Method and Ratio')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "hybridization_heatmap.png"))
        plt.close()