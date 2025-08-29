# phase4/adaptive_nudge.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from .base import DirectedMemoryExperiment

class RecoveryMonitor:
    """Monitors recovery metrics and detects when to trigger nudges"""
    
    def __init__(self, window_size=10, correlation_threshold=0.005, 
                 coherence_threshold=0.98, ccdi_threshold=0.05):
        self.window_size = window_size
        self.correlation_threshold = correlation_threshold
        self.coherence_threshold = coherence_threshold
        self.ccdi_threshold = ccdi_threshold
        
        # Metrics history
        self.correlation_history = []
        self.coherence_history = []
        self.ccdi_history = []
        
    def update(self, metrics):
        """Update metrics history"""
        self.correlation_history.append(metrics['correlation'])
        self.coherence_history.append(metrics['coherence'])
        self.ccdi_history.append(metrics['ccdi'])
        
    def should_trigger_nudge(self):
        """Check if any trigger conditions are met"""
        if len(self.correlation_history) < self.window_size:
            return False, "insufficient_data"
        
        # Check for correlation plateau
        recent_correlation = self.correlation_history[-self.window_size:]
        if len(recent_correlation) >= self.window_size:
            correlation_change = abs(recent_correlation[-1] - recent_correlation[0])
            if correlation_change < self.correlation_threshold:
                return True, "correlation_plateau"
        
        # Check for coherence/correlation imbalance
        if (self.coherence_history[-1] > self.coherence_threshold and 
            self.correlation_history[-1] < 0.8):
            return True, "coherence_imbalance"
        
        # Check for rising CCDI
        if (self.ccdi_history[-1] > self.ccdi_threshold and 
            len(self.ccdi_history) > 1 and 
            self.ccdi_history[-1] > self.ccdi_history[-2]):
            return True, "rising_ccdi"
            
        return False, "no_trigger"

class NudgeGenerator:
    """Generates different types of nudges to apply to the field"""
    
    def __init__(self, experiment):
        self.experiment = experiment
        
    def generate_nudge(self, nudge_type, amplitude=0.05, **kwargs):
        """Generate a nudge based on type"""
        if nudge_type == "amplitude_echo":
            return self._amplitude_echo_nudge(amplitude, **kwargs)
        elif nudge_type == "spatial_bias":
            return self._spatial_bias_nudge(amplitude, **kwargs)
        elif nudge_type == "symmetry_pulse":
            return self._symmetry_pulse_nudge(amplitude, **kwargs)
        else:
            raise ValueError(f"Unknown nudge type: {nudge_type}")
    
    def _amplitude_echo_nudge(self, amplitude, **kwargs):
        """Echo-like nudge that reintroduces a portion of the original pattern"""
        initial_state = self.experiment.initial_state
        current_state = self.experiment.experiment.state
        
        # Create echo nudge (portion of original pattern)
        nudge = amplitude * initial_state
        
        # Apply nudge
        new_state = current_state + nudge
        
        # Normalize to keep values in range
        new_state = np.clip(new_state, -1, 1)
        
        return new_state
        
    def _spatial_bias_nudge(self, amplitude, center=None, radius=10, **kwargs):
        """Spatially biased nudge that affects a specific region"""
        size = self.experiment.experiment.size
        initial_state = self.experiment.initial_state
        current_state = self.experiment.experiment.state
        
        # Default center to middle if not specified
        if center is None:
            center = (size // 2, size // 2)
            
        # Create mask for the specified region
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        mask = ((X - center[0])**2 + (Y - center[1])**2 <= radius**2).astype(float)
        
        # Create spatial nudge
        nudge = amplitude * initial_state * mask
        
        # Apply nudge
        new_state = current_state + nudge
        
        # Normalize
        new_state = np.clip(new_state, -1, 1)
        
        return new_state
    
    def _symmetry_pulse_nudge(self, amplitude, **kwargs):
        """Symmetry nudge that applies mirrored pattern with decay mask"""
        current_state = self.experiment.experiment.state
        
        # Mirror the current state
        mirrored_state = np.flip(current_state, axis=0)
        
        # Create decay mask (higher in center, lower at edges)
        size = self.experiment.experiment.size
        center = size // 2
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt((X - center)**2 + (Y - center)**2)
        max_distance = np.sqrt(2) * center
        decay_mask = 1 - (distance / max_distance)
        
        # Apply symmetry nudge
        nudge = amplitude * mirrored_state * decay_mask
        
        # Apply nudge
        new_state = current_state + nudge
        
        # Normalize
        new_state = np.clip(new_state, -1, 1)
        
        return new_state

class AdaptiveNudgeController(DirectedMemoryExperiment):
    """Controls adaptive nudging based on recovery monitoring"""
    
    def __init__(self, 
                 output_dir="phase4_results/adaptive_nudge",
                 alpha=0.35, 
                 beta=0.5, 
                 gamma=0.92, 
                 pattern_type="fractal",
                 monitor_window=10,
                 max_steps=100,
                 nudge_types=None):
        """Initialize the adaptive nudge controller"""
        super().__init__(output_dir, alpha, beta, gamma, pattern_type)
        
        self.monitor_window = monitor_window
        self.max_steps = max_steps
        self.nudge_types = nudge_types or ["amplitude_echo", "spatial_bias", "symmetry_pulse"]
        
        # Create output subdirectories
        os.makedirs(os.path.join(output_dir, "nudge_events"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        
        # Initialize component objects
        self.monitor = None
        self.nudger = None
        self.results = []
        
    def run(self, num_trials=1, initial_perturbation=True, parallel=False):
        """Run the adaptive nudging experiment"""
        if parallel and num_trials > 1:
            return self._run_parallel(num_trials, initial_perturbation)
        
        all_results = []
        
        for trial in tqdm(range(num_trials), desc="Adaptive Nudge Trials"):
            # Initialize experiment
            self.initialize_experiment()
            
            # Setup monitor and nudger
            self.monitor = RecoveryMonitor(window_size=self.monitor_window)
            self.nudger = NudgeGenerator(self)
            
            # Apply initial perturbation if requested
            if initial_perturbation:
                self.experiment.apply_perturbation(perturbation_type="flip", magnitude=1.0, radius=15)
            
            # Initialize tracking variables
            step = 0
            nudge_events = []
            
            # Run recovery with adaptive nudging
            while step < self.max_steps:
                # Run one step
                self.experiment.update(steps=1)
                step += 1
                
                # Compute metrics
                metrics = self.compute_metrics()
                
                # Update monitor
                self.monitor.update(metrics)
                
                # Check for nudge trigger
                should_nudge, trigger_reason = self.monitor.should_trigger_nudge()
                
                if should_nudge:
                    # Choose a nudge type
                    nudge_type = np.random.choice(self.nudge_types)
                    
                    # Generate and apply nudge
                    amplitude = 0.05  # Could be made adaptive
                    new_state = self.nudger.generate_nudge(nudge_type, amplitude)
                    
                    # Apply the nudge
                    pre_nudge_state = self.experiment.state.copy()
                    pre_nudge_metrics = metrics.copy()
                    
                    # Save snapshot before nudge
                    snapshot_idx = self.save_field_snapshot(f"pre_nudge_step_{step}")
                    
                    # Apply nudge
                    self.experiment.state = new_state.copy()
                    
                    # Update metrics after nudge
                    self.experiment._calculate_metrics()
                    post_nudge_metrics = {
                        'correlation': self.experiment.metrics['correlation'][-1],
                        'coherence': self.experiment.metrics['coherence'][-1],
                        'ccdi': self.experiment.metrics['coherence'][-1] - self.experiment.metrics['correlation'][-1],
                        'spectral_entropy': self.experiment.metrics['spectral_entropy'][-1],
                    }
                    
                    # Record nudge event
                    nudge_event = {
                        'step': step,
                        'trigger': trigger_reason,
                        'nudge_type': nudge_type,
                        'amplitude': amplitude,
                        'pre_correlation': pre_nudge_metrics['correlation'],
                        'post_correlation': post_nudge_metrics['correlation'],
                        'pre_ccdi': pre_nudge_metrics['ccdi'],
                        'post_ccdi': post_nudge_metrics['ccdi'],
                        'impact': post_nudge_metrics['correlation'] - pre_nudge_metrics['correlation']
                    }
                    nudge_events.append(nudge_event)
                    
                    # Save nudge visualization
                    self._save_nudge_visualization(
                        pre_nudge_state, 
                        new_state, 
                        step, 
                        nudge_type, 
                        nudge_event,
                        trial
                    )
                
                # Check for recovery completion
                if metrics['correlation'] > 0.95:
                    break
            
            # Process trial results
            trial_result = {
                'trial_id': trial,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'total_steps': step,
                'final_correlation': metrics['correlation'],
                'final_coherence': metrics['coherence'],
                'final_ccdi': metrics['ccdi'],
                'num_nudges': len(nudge_events),
                'nudge_events': nudge_events
            }
            
            all_results.append(trial_result)
            
            # Save visualization
            self._visualize_trial(trial_result, trial)
        
        # Save and analyze results
        self.results = all_results
        self._analyze_results()
        
        return all_results
    
    def _run_parallel(self, num_trials, initial_perturbation):
        """Run trials in parallel"""
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor() as executor:
            futures = []
            for trial in range(num_trials):
                # Create a new controller for each trial
                controller = AdaptiveNudgeController(
                    output_dir=self.output_dir,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                    pattern_type=self.pattern_type,
                    monitor_window=self.monitor_window,
                    max_steps=self.max_steps,
                    nudge_types=self.nudge_types
                )
                futures.append(
                    executor.submit(controller.run, 1, initial_perturbation, False)
                )
            
            # Collect results
            all_results = []
            for future in tqdm(futures, desc="Collecting results"):
                trial_results = future.result()
                all_results.extend(trial_results)
                
        # Save and analyze results
        self.results = all_results
        self._analyze_results()
        
        return all_results
    
    def _save_nudge_visualization(self, pre_state, post_state, step, nudge_type, nudge_event, trial):
        """Save visualization of a nudge event"""
        plt.figure(figsize=(12, 4))
        
        # Plot the pre-nudge state
        plt.subplot(1, 3, 1)
        plt.imshow(pre_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Pre-Nudge (cor={nudge_event['pre_correlation']:.3f})")
        plt.axis('off')
        
        # Plot the difference
        plt.subplot(1, 3, 2)
        plt.imshow(post_state - pre_state, cmap='RdBu', vmin=-0.1, vmax=0.1)
        plt.title(f"{nudge_type} (Δ={nudge_event['impact']:.4f})")
        plt.axis('off')
        
        # Plot the post-nudge state
        plt.subplot(1, 3, 3)
        plt.imshow(post_state, cmap='viridis', vmin=-1, vmax=1)
        plt.title(f"Post-Nudge (cor={nudge_event['post_correlation']:.3f})")
        plt.axis('off')
        
        plt.suptitle(f"Nudge at Step {step} (Trigger: {nudge_event['trigger']})")
        plt.tight_layout()
        
        # Save the figure
        filename = f"trial{trial}_nudge_step{step}_{nudge_type}.png"
        plt.savefig(os.path.join(self.output_dir, "nudge_events", filename))
        plt.close()
    
    def _visualize_trial(self, trial_result, trial):
        """Create visualization for a trial"""
        plt.figure(figsize=(12, 8))
        
        # Plot correlation curve
        plt.subplot(2, 1, 1)
        plt.plot(self.experiment.metrics['correlation'], 'b-', label='Correlation')
        plt.plot(self.experiment.metrics['coherence'], 'g-', label='Coherence')
        
        # Mark nudge events
        for event in trial_result['nudge_events']:
            plt.axvline(x=event['step'], color='r', linestyle='--', 
                      alpha=0.5, label='Nudge' if event == trial_result['nudge_events'][0] else None)
            plt.scatter(event['step'], event['post_correlation'], color='r', s=50, alpha=0.7)
        
        plt.title(f"Recovery with Adaptive Nudging (Trial {trial})")
        plt.xlabel("Time Step")
        plt.ylabel("Metric Value")
        plt.grid(True)
        plt.legend()
        
        # Plot CCDI and nudge impact
        plt.subplot(2, 1, 2)
        # Calculate CCDI from metrics
        ccdi = [c - r for c, r in zip(self.experiment.metrics['coherence'], 
                                     self.experiment.metrics['correlation'])]
        plt.plot(ccdi, 'k-', label='CCDI')
        
        # Plot nudge impacts
        steps = [event['step'] for event in trial_result['nudge_events']]
        impacts = [event['impact'] for event in trial_result['nudge_events']]
        types = [event['nudge_type'] for event in trial_result['nudge_events']]
        
        type_colors = {
            'amplitude_echo': 'blue',
            'spatial_bias': 'green',
            'symmetry_pulse': 'purple'
        }
        
        for i, (step, impact, nudge_type) in enumerate(zip(steps, impacts, types)):
            color = type_colors.get(nudge_type, 'gray')
            plt.scatter(step, impact, color=color, s=100, alpha=0.7, 
                       label=nudge_type if nudge_type not in [types[:i]] else None)
        
        plt.title("CCDI and Nudge Impacts")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"trial{trial}_recovery_summary.png"
        plt.savefig(os.path.join(self.output_dir, "plots", filename))
        plt.close()
    
    def _analyze_results(self):
        """Analyze and save results summary"""
        if not self.results:
            return
            
        # Extract nudge effectiveness by type
        nudge_impacts = {}
        for trial in self.results:
            for event in trial['nudge_events']:
                nudge_type = event['nudge_type']
                if nudge_type not in nudge_impacts:
                    nudge_impacts[nudge_type] = []
                nudge_impacts[nudge_type].append(event['impact'])
        
        # Calculate average impact by type
        avg_impacts = {}
        for nudge_type, impacts in nudge_impacts.items():
            avg_impacts[nudge_type] = np.mean(impacts) if impacts else 0.0
        
        # Create summary visualization
        plt.figure(figsize=(12, 6))
        
        # Plot nudge count and average impact by type
        types = list(avg_impacts.keys())
        counts = [len(nudge_impacts[t]) for t in types]
        impacts = [avg_impacts[t] for t in types]
        
        # Create a figure with two subplots sharing x axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Nudge counts
        bars = ax1.bar(types, counts, color='blue', alpha=0.7)
        ax1.set_ylabel('Number of Nudges')
        ax1.set_title('Nudge Count by Type')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # Average impacts
        bars = ax2.bar(types, impacts, color='green', alpha=0.7)
        ax2.set_xlabel('Nudge Type')
        ax2.set_ylabel('Average Impact')
        ax2.set_title('Average Impact by Nudge Type')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add impact labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "nudge_type_analysis.png"))
        plt.close()
        
        # Create success rate chart
        success_count = sum(1 for r in self.results if r['final_correlation'] > 0.9)
        success_rate = success_count / len(self.results) if self.results else 0
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Success', 'Failure'], 
               [success_rate, 1 - success_rate],
               color=['green', 'red'],
               alpha=0.7)
        plt.title(f'Recovery Success Rate: {success_rate:.1%}')
        plt.ylabel('Percentage')
        plt.ylim(0, 1)
        
        # Add percentage labels
        plt.text(0, success_rate + 0.02, f'{success_rate:.1%}', 
                ha='center', va='bottom')
        plt.text(1, (1-success_rate) + 0.02, f'{1-success_rate:.1%}', 
                ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.output_dir, "success_rate.png"))
        plt.close()
        
        # Save summary to file
        summary = {
            'total_trials': len(self.results),
            'success_rate': success_rate,
            'avg_nudges_per_trial': np.mean([r['num_nudges'] for r in self.results]),
            'nudge_type_impacts': avg_impacts,
            'nudge_counts': {t: len(impacts) for t, impacts in nudge_impacts.items()}
        }
        
        with open(os.path.join(self.output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
    
    def visualize_results(self, **kwargs):
        """Create comprehensive visualization of results"""
        # This will create a more detailed analysis visualization
        # Implementation would depend on the specific analysis needs
        pass