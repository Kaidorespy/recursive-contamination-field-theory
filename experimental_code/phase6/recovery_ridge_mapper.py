# Create a new file: phase6/recovery_ridge_mapper.py

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from rcft_metrics import compute_ccdi

class RecoveryRidgeMapper:
    """Maps the recovery landscape of hybrid memories to find switchable vs. locked states"""
    
    def __init__(self, phase6):
        """
        Initialize the recovery ridge mapper
        
        Parameters:
        -----------
        phase6 : Phase6MultiMemory
            Reference to the main Phase VI object
        """
        self.phase6 = phase6
        self.logger = phase6.logger
        self.output_dir = os.path.join(phase6.output_dir, "recovery_ridge")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_recovery_ridge_experiment(self, pattern_ids, blend_method='pixel', 
                                   bias_type='none', cue_strength=0.5, blend_ratio=0.5):
        """
        Run the recovery ridge mapping experiment
        
        Parameters:
        -----------
        pattern_ids : list
            List of pattern IDs to blend and test
        blend_method : str
            Method for blending patterns: 'pixel', 'structured', or 'frequency'
        bias_type : str
            Type of bias to apply: 'none', 'fingerprint', 'gradient', or 'context'
        cue_strength : float
            Strength of context cue (0.0 to 1.0)
        blend_ratio : float
            Ratio for blending patterns (0.0 to 1.0)
            
        Returns:
        --------
        pd.DataFrame
            Results of the recovery ridge mapping
        """
        if len(pattern_ids) < 2:
            raise ValueError("Need at least two patterns for recovery ridge mapping")
            
        self.logger.info(f"Running recovery ridge mapping with patterns: {pattern_ids}")
        self.logger.info(f"Blend method: {blend_method}, Bias type: {bias_type}, Cue strength: {cue_strength}")
        
        # Get patterns
        patterns = []
        for pid in pattern_ids:
            if pid not in self.phase6.memory_bank:
                raise ValueError(f"Pattern {pid} not found in memory bank")
            patterns.append(self.phase6.memory_bank[pid])
            
        # Create hybrid pattern
        hybrid_pattern = self.phase6.blender.blend_patterns(
            patterns[0].pattern_id, 
            patterns[1].pattern_id, 
            ratio=blend_ratio, 
            method=blend_method
        )
        
        # Create hybrid pattern ID
        hybrid_id = f"{patterns[0].pattern_id}_{patterns[1].pattern_id}_{blend_ratio:.2f}_{blend_method}"
        
        # Run recovery trajectory mapping
        trajectories = self._map_recovery_trajectories(
            hybrid_pattern, hybrid_id, patterns, bias_type, cue_strength
        )
        
        # Create recovery landscape visualization
        self._visualize_recovery_landscape(
            trajectories, hybrid_id, patterns, bias_type, cue_strength
        )
        
        # Run switchability analysis
        switchability = self._analyze_switchability(
            trajectories, hybrid_id, patterns, bias_type, cue_strength
        )
        
        # Combine results
        results = {
            'hybrid_id': hybrid_id,
            'pattern_a': patterns[0].pattern_id,
            'pattern_b': patterns[1].pattern_id,
            'blend_method': blend_method,
            'bias_type': bias_type,
            'cue_strength': cue_strength,
            'blend_ratio': blend_ratio,
            'switchability': switchability,
            'trajectories': trajectories
        }
        
        # Save results to CSV
        results_df = pd.DataFrame([{
            'hybrid_id': hybrid_id,
            'pattern_a': patterns[0].pattern_id,
            'pattern_b': patterns[1].pattern_id,
            'blend_method': blend_method,
            'bias_type': bias_type,
            'cue_strength': cue_strength,
            'blend_ratio': blend_ratio,
            'switchability_a': switchability['to_a'],
            'switchability_b': switchability['to_b'],
            'recovery_speed_a': switchability['recovery_speed_a'],
            'recovery_speed_b': switchability['recovery_speed_b'],
            'ridge_width': switchability['ridge_width'],
            'ridge_stability': switchability['ridge_stability']
        }])
        results_df.to_csv(os.path.join(self.output_dir, "recovery_ridge_results.csv"), index=False)
        
        return results_df
    
    def _map_recovery_trajectories(self, hybrid_pattern, hybrid_id, base_patterns, bias_type, cue_strength):
        """
        Map recovery trajectories for hybrid pattern with different perturbation types
        
        Parameters:
        -----------
        hybrid_pattern : ndarray
            The hybrid pattern to test
        hybrid_id : str
            ID for the hybrid pattern
        base_patterns : list
            List of base pattern memory traces
        bias_type : str
            Type of bias to apply
        cue_strength : float
            Strength of context cue
            
        Returns:
        --------
        dict
            Dictionary of trajectory data
        """
        # Setup experiment
        exp = self.phase6.base_experiment
        
        # Define perturbation grid
        perturbation_types = ["flip", "noise", "zero"]
        perturbation_magnitudes = [0.2, 0.5, 0.8, 1.2, 1.5]
        
        # Store trajectories
        trajectories = {
            'types': perturbation_types,
            'magnitudes': perturbation_magnitudes,
            'data': {}
        }
        
        # For each perturbation type and magnitude
        for pert_type in perturbation_types:
            trajectories['data'][pert_type] = {}
            
            for pert_mag in perturbation_magnitudes:
                # Initialize with hybrid pattern
                exp.state = hybrid_pattern.copy()
                
                # Let it stabilize briefly to find initial attractor
                exp.update(steps=20)
                initial_state = exp.state.copy()
                
                # Apply perturbation
                exp.apply_perturbation(perturbation_type=pert_type, magnitude=pert_mag)
                perturbed_state = exp.state.copy()
                
                # Apply bias if specified
                if bias_type != 'none':
                    # Apply bias based on type
                    biased_state = self._apply_bias(
                        perturbed_state, 
                        base_patterns, 
                        bias_type, 
                        cue_strength
                    )
                    exp.state = biased_state.copy()
                    
                # Record correlation series
                correlation_a = []
                correlation_b = []
                steps = []
                states = []
                
                # Run recovery for 50 steps, recording every step
                for step in range(50):
                    # Update for one step
                    exp.update(steps=1)
                    
                    # Calculate correlation to base patterns
                    corr_a = np.corrcoef(base_patterns[0].initial_state.flatten(), 
                                      exp.state.flatten())[0, 1]
                    corr_b = np.corrcoef(base_patterns[1].initial_state.flatten(), 
                                      exp.state.flatten())[0, 1]
                    
                    correlation_a.append(corr_a)
                    correlation_b.append(corr_b)
                    steps.append(step)
                    states.append(exp.state.copy())
                
                # Store trajectory data
                trajectories['data'][pert_type][pert_mag] = {
                    'initial_state': initial_state,
                    'perturbed_state': perturbed_state,
                    'final_state': exp.state.copy(),
                    'correlation_a': correlation_a,
                    'correlation_b': correlation_b,
                    'steps': steps,
                    'states': states
                }
                
                # Save states
                np.save(os.path.join(self.output_dir, 
                                  f"{hybrid_id}_{pert_type}_{pert_mag:.1f}_initial.npy"), 
                      initial_state)
                np.save(os.path.join(self.output_dir, 
                                  f"{hybrid_id}_{pert_type}_{pert_mag:.1f}_perturbed.npy"), 
                      perturbed_state)
                np.save(os.path.join(self.output_dir, 
                                  f"{hybrid_id}_{pert_type}_{pert_mag:.1f}_final.npy"), 
                      exp.state.copy())
                
                # Create individual trajectory visualization
                self._visualize_trajectory(
                    hybrid_id, pert_type, pert_mag, 
                    base_patterns, 
                    initial_state, perturbed_state, exp.state.copy(),
                    correlation_a, correlation_b, steps
                )
                
                self.logger.info(f"Mapped trajectory for {pert_type} (mag={pert_mag:.1f}) - " +
                              f"Final correlations: A={correlation_a[-1]:.3f}, B={correlation_b[-1]:.3f}")
        
        return trajectories
    
    def _apply_bias(self, state, base_patterns, bias_type, strength):
        """
        Apply recovery bias to the state
        
        Parameters:
        -----------
        state : ndarray
            State to apply bias to
        base_patterns : list
            List of base pattern memory traces
        bias_type : str
            Type of bias to apply
        strength : float
            Strength of bias
            
        Returns:
        --------
        ndarray
            Biased state
        """
        # Make a copy of the state
        biased_state = state.copy()
        
        # Based on bias type, modify the state
        if bias_type == 'fingerprint':
            # Apply fingerprint bias (selective feature enhancement)
            # Get the patterns' fingerprints
            for pattern in base_patterns:
                if pattern.fingerprint is None:
                    pattern.compute_fingerprint()
            
            # Currently, we'll just do a simple bias toward one pattern or the other
            # based on the current state's similarity to each pattern
            corr_a = np.corrcoef(base_patterns[0].initial_state.flatten(), 
                               state.flatten())[0, 1]
            corr_b = np.corrcoef(base_patterns[1].initial_state.flatten(), 
                               state.flatten())[0, 1]
            
            # Determine which pattern to bias toward (the closer one)
            if corr_a > corr_b:
                # Bias toward pattern A
                target = base_patterns[0].initial_state
            else:
                # Bias toward pattern B
                target = base_patterns[1].initial_state
            
            # Apply subtle bias by enhancing correlations
            diff = target - state
            biased_state = state + strength * 0.2 * diff
            
        elif bias_type == 'gradient':
            # Apply gradient bias (spatial gradient of influence)
            pattern_a = base_patterns[0].initial_state
            pattern_b = base_patterns[1].initial_state
            
            # Create a gradient mask
            x = np.linspace(0, 1, state.shape[0])
            y = np.linspace(0, 1, state.shape[1])
            X, Y = np.meshgrid(x, y)
            
            # Use a diagonal gradient
            mask = X
            
            # Apply mask to blend the two patterns' influence
            diff_a = pattern_a - state
            diff_b = pattern_b - state
            
            biased_state = state + strength * (mask * diff_a + (1 - mask) * diff_b)
            
        elif bias_type == 'context':
            # Apply context bias (partial pattern cue)
            # This is similar to what the ContextSwitcher does
            
            # Create two biased versions - one toward each pattern
            biased_a = self._apply_context_cue(state, base_patterns[0].initial_state, 
                                             strength)
            biased_b = self._apply_context_cue(state, base_patterns[1].initial_state, 
                                             strength)
            
            # Determine which is closer to the current state
            diff_a = np.mean(np.abs(biased_a - state))
            diff_b = np.mean(np.abs(biased_b - state))
            
            # Use the closer one
            if diff_a < diff_b:
                biased_state = biased_a
            else:
                biased_state = biased_b
            
        # Normalize to [-1, 1]
        state_min = np.min(biased_state)
        state_max = np.max(biased_state)
        if state_max > state_min:  # Avoid division by zero
            normalized_state = 2.0 * (biased_state - state_min) / (state_max - state_min) - 1.0
            return normalized_state
        
        return biased_state
    
    def _apply_context_cue(self, source_state, target_state, cue_strength):
        """Apply a context cue by blending source and target in a spatial region"""
        # Make a copy of source state
        result = source_state.copy()
        
        # Create circular region of influence
        mask = np.zeros_like(source_state, dtype=bool)
        center = (source_state.shape[0] // 2, source_state.shape[1] // 2)
        radius = int(cue_strength * source_state.shape[0] // 2)  # Size scales with cue strength
        
        x = np.arange(source_state.shape[0])
        y = np.arange(source_state.shape[1])
        X, Y = np.meshgrid(x, y)
        mask = ((X - center[0])**2 + (Y - center[1])**2 <= radius**2)
        
        # Apply mask
        result[mask] = target_state[mask]
            
        return result
    
    def _visualize_trajectory(self, hybrid_id, pert_type, pert_mag, 
                           base_patterns, initial_state, perturbed_state, final_state,
                           correlation_a, correlation_b, steps):
        """
        Create visualization for a single recovery trajectory
        
        Parameters:
        -----------
        hybrid_id : str
            ID for the hybrid pattern
        pert_type : str
            Perturbation type
        pert_mag : float
            Perturbation magnitude
        base_patterns : list
            List of base pattern memory traces
        initial_state, perturbed_state, final_state : ndarray
            States at different points in the trajectory
        correlation_a, correlation_b : list
            Correlation series to each base pattern
        steps : list
            Time steps
        """
        plt.figure(figsize=(15, 10))
        
        # Plot initial, perturbed, and final states
        plt.subplot(2, 3, 1)
        plt.imshow(initial_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Initial State\n{hybrid_id}")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(perturbed_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"After {pert_type.title()} Perturbation\nMagnitude: {pert_mag:.1f}")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(final_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Final State")
        plt.colorbar()
        plt.axis('off')
        
        # Plot correlation time series
        plt.subplot(2, 1, 2)
        plt.plot(steps, correlation_a, 'b-', label=f'Correlation to {base_patterns[0].pattern_id}')
        plt.plot(steps, correlation_b, 'g-', label=f'Correlation to {base_patterns[1].pattern_id}')
        
        # Add line at crossover point if it exists
        crossover = -1
        for i in range(1, len(steps)):
            if (correlation_a[i-1] < correlation_b[i-1] and correlation_a[i] >= correlation_b[i]) or \
               (correlation_a[i-1] > correlation_b[i-1] and correlation_a[i] <= correlation_b[i]):
                crossover = steps[i]
                plt.axvline(x=crossover, color='r', linestyle='--', 
                          label=f'Crossover at step {crossover}')
                break
        
        plt.title(f"Recovery Trajectory - {hybrid_id} - {pert_type} ({pert_mag:.1f})")
        plt.xlabel("Time Step")
        plt.ylabel("Correlation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotation text 
        final_corr_a = correlation_a[-1]
        final_corr_b = correlation_b[-1]
        dominant = base_patterns[0].pattern_id if final_corr_a > final_corr_b else base_patterns[1].pattern_id
        
        plt.figtext(0.01, 0.01, 
                  f"Final correlations: {base_patterns[0].pattern_id}={final_corr_a:.3f}, " +
                  f"{base_patterns[1].pattern_id}={final_corr_b:.3f}\n" +
                  f"Dominant pattern: {dominant}\n" +
                  f"Recovery bias: {abs(final_corr_a - final_corr_b):.3f}",
                  fontsize=10, backgroundcolor='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                              f"{hybrid_id}_{pert_type}_{pert_mag:.1f}_trajectory.png"))
        plt.close()
    
    # Create a completely new visualization method using a simpler approach

    def _visualize_recovery_landscape(self, trajectories, hybrid_id, base_patterns, bias_type, cue_strength):
        """
        Create recovery landscape visualization
        
        Parameters:
        -----------
        trajectories : dict
            Dictionary of trajectory data
        hybrid_id : str
            ID for the hybrid pattern
        base_patterns : list
            List of base pattern memory traces
        bias_type : str
            Type of bias applied
        cue_strength : float
            Strength of context cue
        """
        # Create a heatmap of final correlations
        plt.figure(figsize=(15, 10))
        
        # Initialize arrays for heatmap data
        n_types = len(trajectories['types'])
        n_mags = len(trajectories['magnitudes'])
        
        corr_diff_map = np.zeros((n_types, n_mags))
        corr_a_map = np.zeros((n_types, n_mags))
        corr_b_map = np.zeros((n_types, n_mags))
        recovery_time_map = np.zeros((n_types, n_mags))
        
        # Fill in the data
        for i, pert_type in enumerate(trajectories['types']):
            for j, pert_mag in enumerate(trajectories['magnitudes']):
                # Get trajectory data
                traj = trajectories['data'][pert_type][pert_mag]
                
                # Calculate correlation difference
                corr_a = traj['correlation_a'][-1]
                corr_b = traj['correlation_b'][-1]
                
                corr_diff_map[i, j] = corr_a - corr_b
                corr_a_map[i, j] = corr_a
                corr_b_map[i, j] = corr_b
                
                # Calculate recovery time (steps until stabilization)
                # Defined as when correlation stops changing significantly
                recovery_step = len(traj['steps']) - 1  # Default to last step
                
                for step in range(5, len(traj['steps']) - 5):
                    # Check if correlation is stable for next 5 steps
                    corr_window_a = traj['correlation_a'][step:step+5]
                    corr_window_b = traj['correlation_b'][step:step+5]
                    
                    if np.max(corr_window_a) - np.min(corr_window_a) < 0.01 and \
                    np.max(corr_window_b) - np.min(corr_window_b) < 0.01:
                        recovery_step = step
                        break
                
                recovery_time_map[i, j] = recovery_step
        
        # Simple approach with matplotlib instead of seaborn
        # Plot correlation difference map
        ax1 = plt.subplot(2, 2, 1)
        img1 = ax1.imshow(corr_diff_map, cmap='coolwarm')
        plt.colorbar(img1, ax=ax1, label=f"Correlation Diff ({base_patterns[0].pattern_id} - {base_patterns[1].pattern_id})")
        ax1.set_title(f"Recovery Landscape: {base_patterns[0].pattern_id} vs. {base_patterns[1].pattern_id}")
        ax1.set_xticks(np.arange(len(trajectories['magnitudes'])))
        ax1.set_xticklabels([f"{m:.1f}" for m in trajectories['magnitudes']])
        ax1.set_yticks(np.arange(len(trajectories['types'])))
        ax1.set_yticklabels(trajectories['types'])
        ax1.set_xlabel("Perturbation Magnitude")
        ax1.set_ylabel("Perturbation Type")
        
        # Plot correlation to pattern A
        ax2 = plt.subplot(2, 2, 2)
        img2 = ax2.imshow(corr_a_map, cmap='Blues')
        plt.colorbar(img2, ax=ax2, label=f"Correlation to {base_patterns[0].pattern_id}")
        ax2.set_title(f"Correlation to {base_patterns[0].pattern_id}")
        ax2.set_xticks(np.arange(len(trajectories['magnitudes'])))
        ax2.set_xticklabels([f"{m:.1f}" for m in trajectories['magnitudes']])
        ax2.set_yticks(np.arange(len(trajectories['types'])))
        ax2.set_yticklabels(trajectories['types'])
        ax2.set_xlabel("Perturbation Magnitude")
        ax2.set_ylabel("Perturbation Type")
        
        # Plot correlation to pattern B
        ax3 = plt.subplot(2, 2, 3)
        img3 = ax3.imshow(corr_b_map, cmap='Greens')
        plt.colorbar(img3, ax=ax3, label=f"Correlation to {base_patterns[1].pattern_id}")
        ax3.set_title(f"Correlation to {base_patterns[1].pattern_id}")
        ax3.set_xticks(np.arange(len(trajectories['magnitudes'])))
        ax3.set_xticklabels([f"{m:.1f}" for m in trajectories['magnitudes']])
        ax3.set_yticks(np.arange(len(trajectories['types'])))
        ax3.set_yticklabels(trajectories['types'])
        ax3.set_xlabel("Perturbation Magnitude")
        ax3.set_ylabel("Perturbation Type")
        
        # Plot recovery time map
        ax4 = plt.subplot(2, 2, 4)
        img4 = ax4.imshow(recovery_time_map, cmap='YlOrRd')
        plt.colorbar(img4, ax=ax4, label=f"Steps to Stabilization")
        ax4.set_title(f"Recovery Time")
        ax4.set_xticks(np.arange(len(trajectories['magnitudes'])))
        ax4.set_xticklabels([f"{m:.1f}" for m in trajectories['magnitudes']])
        ax4.set_yticks(np.arange(len(trajectories['types'])))
        ax4.set_yticklabels(trajectories['types'])
        ax4.set_xlabel("Perturbation Magnitude")
        ax4.set_ylabel("Perturbation Type")
        
        # Add overall title
        plt.suptitle(f"Recovery Ridge Map: {hybrid_id}\nBias: {bias_type}, Cue Strength: {cue_strength:.2f}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, f"{hybrid_id}_recovery_landscape.png"))
        plt.close()
    
    def _visualize_3d_landscape(self, trajectories, hybrid_id, base_patterns, 
                            bias_type, cue_strength,
                            corr_diff_map, recovery_time_map,
                            custom_cmap=None):
        """
        Create 3D visualization of recovery landscape
        
        Parameters:
        -----------
        trajectories : dict
            Dictionary of trajectory data
        hybrid_id : str
            ID for the hybrid pattern
        base_patterns : list
            List of base pattern memory traces
        bias_type : str
            Type of bias applied
        cue_strength : float
            Strength of context cue
        corr_diff_map : ndarray
            Map of correlation differences
        recovery_time_map : ndarray
            Map of recovery times
        custom_cmap : matplotlib colormap, optional
            Custom colormap for visualization
        """
        # Create a 3D visualization
        from mpl_toolkits.mplot3d import Axes3D
        
        # Prepare data for 3D surface
        X, Y = np.meshgrid(
            np.arange(len(trajectories['magnitudes'])), 
            np.arange(len(trajectories['types']))
        )
        
        # Get the X, Y coordinates of the ridge (where corr_diff_map is closest to 0)
        ridge_mask = np.abs(corr_diff_map) < 0.1
        ridge_y, ridge_x = np.where(ridge_mask)
        
        # Create the 3D plot
        fig = plt.figure(figsize=(15, 10))
        
        # Use the provided colormap or create a default diverging one
        if custom_cmap is None:
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_div', 
                [(0, 'blue'), (0.5, 'white'), (1, 'green')]
            )
        
        # Plot correlation difference landscape
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, corr_diff_map, cmap=custom_cmap, alpha=0.8)
        
        # Highlight the ridge
        if len(ridge_x) > 0 and len(ridge_y) > 0:
            ax1.scatter(ridge_x, ridge_y, corr_diff_map[ridge_y, ridge_x], 
                    color='r', s=50, label='Recovery Ridge')
        
        ax1.set_xlabel("Perturbation Magnitude")
        ax1.set_ylabel("Perturbation Type")
        ax1.set_zlabel(f"Correlation Diff ({base_patterns[0].pattern_id} - {base_patterns[1].pattern_id})")
        ax1.set_title(f"Recovery Landscape")
        
        ax1.set_xticks(np.arange(len(trajectories['magnitudes'])))
        ax1.set_xticklabels([f"{m:.1f}" for m in trajectories['magnitudes']])
        ax1.set_yticks(np.arange(len(trajectories['types'])))
        ax1.set_yticklabels(trajectories['types'])
        
        # Plot recovery time landscape
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, recovery_time_map, cmap='YlOrRd', alpha=0.8)
        
        # Highlight the ridge on recovery time surface
        if len(ridge_x) > 0 and len(ridge_y) > 0:
            ax2.scatter(ridge_x, ridge_y, recovery_time_map[ridge_y, ridge_x], 
                    color='b', s=50, label='Recovery Ridge')
        
        ax2.set_xlabel("Perturbation Magnitude")
        ax2.set_ylabel("Perturbation Type")
        ax2.set_zlabel("Recovery Time (steps)")
        ax2.set_title(f"Recovery Time Landscape")
        
        ax2.set_xticks(np.arange(len(trajectories['magnitudes'])))
        ax2.set_xticklabels([f"{m:.1f}" for m in trajectories['magnitudes']])
        ax2.set_yticks(np.arange(len(trajectories['types'])))
        ax2.set_yticklabels(trajectories['types'])
        
        # Add overall title
        plt.suptitle(f"3D Recovery Ridge Map: {hybrid_id}\nBias: {bias_type}, Cue Strength: {cue_strength:.2f}", 
                fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, f"{hybrid_id}_3d_landscape.png"))
        plt.close()
    
    def _analyze_switchability(self, trajectories, hybrid_id, base_patterns, bias_type, cue_strength):
        """
        Analyze the switchability of the hybrid state
        
        Parameters:
        -----------
        trajectories : dict
            Dictionary of trajectory data
        hybrid_id : str
            ID for the hybrid pattern
        base_patterns : list
            List of base pattern memory traces
        bias_type : str
            Type of bias applied
        cue_strength : float
            Strength of context cue
            
        Returns:
        --------
        dict
            Switchability metrics
        """
        # Calculate metrics
        # Count cases where recovery goes to each pattern
        count_to_a = 0
        count_to_b = 0
        total_cases = 0
        
        # Sum of recovery speeds
        recovery_steps_a = []
        recovery_steps_b = []
        
        # Find the ridge (where correlation diff is close to 0)
        ridge_points = []
        
        # Process all trajectories
        for pert_type in trajectories['types']:
            for pert_mag in trajectories['magnitudes']:
                # Get trajectory data
                traj = trajectories['data'][pert_type][pert_mag]
                
                # Calculate correlation difference
                corr_a = traj['correlation_a'][-1]
                corr_b = traj['correlation_b'][-1]
                
                total_cases += 1
                
                # Determine which pattern it recovers to
                if corr_a > corr_b:
                    count_to_a += 1
                    
                    # Find recovery speed (steps until correlation to A > 0.7)
                    recovery_step = len(traj['steps']) - 1  # Default to last step
                    for step, corr in enumerate(traj['correlation_a']):
                        if corr > 0.7:
                            recovery_step = step
                            break
                    
                    recovery_steps_a.append(recovery_step)
                else:
                    count_to_b += 1
                    
                    # Find recovery speed (steps until correlation to B > 0.7)
                    recovery_step = len(traj['steps']) - 1  # Default to last step
                    for step, corr in enumerate(traj['correlation_b']):
                        if corr > 0.7:
                            recovery_step = step
                            break
                    
                    recovery_steps_b.append(recovery_step)
                
                # Check if this is a ridge point (correlation diff close to 0)
                if abs(corr_a - corr_b) < 0.1:
                    ridge_points.append({
                        'pert_type': pert_type,
                        'pert_mag': pert_mag,
                        'corr_diff': corr_a - corr_b,
                        'recovery_step': recovery_step
                    })
        
        # Calculate switchability metrics
        switchability_to_a = count_to_a / total_cases if total_cases > 0 else 0.0
        switchability_to_b = count_to_b / total_cases if total_cases > 0 else 0.0
        
        # Calculate recovery speeds
        recovery_speed_a = np.mean(recovery_steps_a) if recovery_steps_a else float('inf')
        recovery_speed_b = np.mean(recovery_steps_b) if recovery_steps_b else float('inf')
        
        # Calculate ridge properties
        ridge_width = len(ridge_points) / total_cases if total_cases > 0 else 0.0
        ridge_stability = 0.0
        if ridge_points:
            # Calculate stability as inverse of average recovery time on the ridge
            ridge_recovery_times = [p['recovery_step'] for p in ridge_points]
            ridge_stability = 1.0 / np.mean(ridge_recovery_times) if ridge_recovery_times else 0.0
        
        # Create a visual summary of switchability
        self._visualize_switchability(
            hybrid_id, base_patterns, 
            bias_type, cue_strength,
            switchability_to_a, switchability_to_b,
            recovery_speed_a, recovery_speed_b,
            ridge_width, ridge_stability,
            ridge_points
        )
        
        return {
            'to_a': switchability_to_a,
            'to_b': switchability_to_b,
            'recovery_speed_a': recovery_speed_a,
            'recovery_speed_b': recovery_speed_b,
            'ridge_width': ridge_width,
            'ridge_stability': ridge_stability,
            'ridge_points': ridge_points
        }
    
    def _visualize_switchability(self, hybrid_id, base_patterns, 
                              bias_type, cue_strength,
                              switchability_to_a, switchability_to_b,
                              recovery_speed_a, recovery_speed_b,
                              ridge_width, ridge_stability,
                              ridge_points):
        """
        Create visualization summarizing switchability
        
        Parameters:
        -----------
        hybrid_id : str
            ID for the hybrid pattern
        base_patterns : list
            List of base pattern memory traces
        bias_type : str
            Type of bias applied
        cue_strength : float
            Strength of context cue
        switchability_to_a, switchability_to_b : float
            Probability of switching to each pattern
        recovery_speed_a, recovery_speed_b : float
            Average steps to recover to each pattern
        ridge_width, ridge_stability : float
            Properties of the recovery ridge
        ridge_points : list
            List of points on the recovery ridge
        """
        plt.figure(figsize=(12, 8))
        
        # Create bar chart of switchability
        plt.subplot(2, 2, 1)
        plt.bar([base_patterns[0].pattern_id, base_patterns[1].pattern_id], 
              [switchability_to_a, switchability_to_b])
        plt.title("Recovery Probability")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        
        # Create bar chart of recovery speed
        plt.subplot(2, 2, 2)
        plt.bar([base_patterns[0].pattern_id, base_patterns[1].pattern_id], 
              [recovery_speed_a, recovery_speed_b])
        plt.title("Recovery Speed (steps)")
        plt.ylabel("Steps to Recover")
        
        # Create bar chart of ridge properties
        plt.subplot(2, 2, 3)
        plt.bar(['Ridge Width', 'Ridge Stability'], [ridge_width, ridge_stability])
        plt.title("Recovery Ridge Properties")
        
        # Create text summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Determine stability type
        if switchability_to_a > 0.8:
            stability = f"Stable to {base_patterns[0].pattern_id}"
        elif switchability_to_b > 0.8:
            stability = f"Stable to {base_patterns[1].pattern_id}" 
        elif ridge_width > 0.3:
            stability = "Meta-stable (ridge)"
        else:
            stability = "Switchable"
            
        summary_text = [
            f"Hybrid Pattern: {hybrid_id}",
            f"Bias: {bias_type}, Cue Strength: {cue_strength:.2f}",
            f"",
            f"Stability Type: {stability}",
            f"",
            f"Switchability to {base_patterns[0].pattern_id}: {switchability_to_a:.2f}",
            f"Switchability to {base_patterns[1].pattern_id}: {switchability_to_b:.2f}",
            f"Recovery Speed to {base_patterns[0].pattern_id}: {recovery_speed_a:.1f} steps",
            f"Recovery Speed to {base_patterns[1].pattern_id}: {recovery_speed_b:.1f} steps",
            f"",
            f"Ridge Width: {ridge_width:.2f}",
            f"Ridge Stability: {ridge_stability:.3f}"
        ]
        
        plt.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12)
        
        # Add title
        plt.suptitle(f"Switchability Analysis: {hybrid_id}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, f"{hybrid_id}_switchability.png"))
        plt.close()
