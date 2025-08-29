"""
Visualization utilities for Phase V of the RCFT framework.
This module provides standardized visualization tools for RCFT experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Set default style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Define custom colormaps for different visualization scenarios
# True memory (saturated) vs false memory (desaturated) vs catastrophic (red)
TRUE_MEMORY_CMAP = sns.color_palette("viridis", as_cmap=True)
FALSE_MEMORY_CMAP = sns.color_palette("Greys", as_cmap=True)
CATASTROPHIC_CMAP = sns.color_palette("Reds", as_cmap=True)

class RCFTVisualizer:
    """Standardized visualizations for RCFT experiments."""
    
    def __init__(self, output_dir="visualization_output", cmap="viridis"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            cmap: Default colormap for visualizations
        """
        self.output_dir = output_dir
        self.cmap = cmap
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_state(self, state, title=None, colorbar=True, vmin=-1, vmax=1, 
                       ax=None, cmap=None, save_path=None):
        """
        Visualize a single state of the field.
        
        Args:
            state: 2D array representing field state
            title: Optional title
            colorbar: Whether to show colorbar
            vmin, vmax: Value limits for colormap
            ax: Optional matplotlib axis to plot on
            cmap: Optional colormap override
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure and axis if ax is None
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure
        
        # Use default cmap if none provided
        cmap = cmap or self.cmap
        
        # Create the plot
        im = ax.imshow(state, cmap=cmap, vmin=vmin, vmax=vmax)
        
        if title:
            ax.set_title(title)
            
        ax.set_axis_off()
        
        if colorbar:
            plt.colorbar(im, ax=ax)
            
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if created_fig:
            return fig, ax
            
    def visualize_state_sequence(self, states, titles=None, ncols=3, title=None, 
                               vmin=-1, vmax=1, cmap=None, save_path=None):
        """
        Visualize a sequence of states.
        
        Args:
            states: List of 2D arrays
            titles: Optional list of titles for each state
            ncols: Number of columns in the grid
            title: Optional overall title
            vmin, vmax: Value limits for colormap
            cmap: Optional colormap override
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_states = len(states)
        nrows = (n_states + ncols - 1) // ncols  # Ceiling division
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Use default cmap if none provided
        cmap = cmap or self.cmap
        
        for i, state in enumerate(states):
            if i < len(axes):
                self.visualize_state(
                    state, 
                    title=titles[i] if titles and i < len(titles) else f"State {i+1}",
                    colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                    ax=axes[i],
                    cmap=cmap
                )
                
        # Hide empty subplots
        for i in range(n_states, len(axes)):
            axes[i].axis('off')
            
        # Add a single colorbar for the entire figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        
        if title:
            fig.suptitle(title, fontsize=16)
            
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_state_animation(self, states, interval=100, title=None, 
                                vmin=-1, vmax=1, cmap=None, save_path=None):
        """
        Create an animation of state evolution.
        
        Args:
            states: List of 2D arrays
            interval: Time interval between frames in milliseconds
            title: Optional title
            vmin, vmax: Value limits for colormap
            cmap: Optional colormap override
            save_path: Optional path to save animation
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use default cmap if none provided
        cmap = cmap or self.cmap
        
        # Initialize with the first state
        im = ax.imshow(states[0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        
        if title:
            title_obj = ax.set_title(title)
            
        # Create animation function
        def update(frame):
            im.set_array(states[frame])
            if title:
                title_obj.set_text(f"{title} - Frame {frame+1}/{len(states)}")
            return [im]
            
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(states), interval=interval, blit=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000/interval)
            else:
                anim.save(save_path, writer='ffmpeg', fps=1000/interval)
                
        plt.close(fig)
        return anim
        
    def visualize_metrics(self, metrics, x_values=None, title=None, 
                        xlabel='Steps', save_path=None):
        """
        Visualize multiple metrics over time.
        
        Args:
            metrics: Dictionary of metric names and arrays of values
            x_values: Optional x-axis values (default: array indices)
            title: Optional title
            xlabel: Label for x-axis
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if x_values is None:
            # Use the length of the first metric array to determine x values
            first_metric = list(metrics.values())[0]
            x_values = np.arange(len(first_metric))
            
        for name, values in metrics.items():
            # Ensure values align with x_values
            if len(values) != len(x_values):
                continue
                
            ax.plot(x_values, values, 'o-', label=name)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Value')
        
        if title:
            ax.set_title(title)
            
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_comparative_metrics(self, metrics_sets, labels=None, x_values=None,
                                    title=None, xlabel='Steps', save_path=None):
        """
        Compare multiple sets of metrics from different runs.
        
        Args:
            metrics_sets: List of dictionaries, each with metric names and arrays
            labels: Optional list of labels for each metrics set
            x_values: Optional x-axis values (default: array indices)
            title: Optional title
            xlabel: Label for x-axis
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not metrics_sets:
            return None
            
        # Get all unique metric names across all sets
        metric_names = set()
        for metrics in metrics_sets:
            metric_names.update(metrics.keys())
            
        metric_names = sorted(list(metric_names))
        n_metrics = len(metric_names)
        
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(metrics_sets))]
            
        if x_values is None:
            # Use the length of the first available metric array
            for metrics in metrics_sets:
                if metrics:
                    first_metric = list(metrics.values())[0]
                    x_values = np.arange(len(first_metric))
                    break
        
        # Calculate grid dimensions
        nrows = (n_metrics + 1) // 2  # Ceiling division
        ncols = min(2, n_metrics)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each metric on its own subplot
        for i, metric_name in enumerate(metric_names):
            if i < len(axes):
                ax = axes[i]
                
                for j, (metrics, label) in enumerate(zip(metrics_sets, labels)):
                    if metric_name in metrics:
                        values = metrics[metric_name]
                        
                        # Ensure values align with x_values
                        if len(values) == len(x_values):
                            ax.plot(x_values, values, 'o-', label=label)
                
                ax.set_title(metric_name)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Value')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
            
        if title:
            fig.suptitle(title, fontsize=16)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_heatmap(self, data, row_labels, col_labels, title=None, 
                        xlabel=None, ylabel=None, cmap="viridis", 
                        annotate=True, save_path=None):
        """
        Create a heatmap visualization.
        
        Args:
            data: 2D array of values
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Optional title
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            cmap: Colormap
            annotate: Whether to show data values in cells
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(max(6, len(col_labels)*0.8 + 2), 
                                      max(6, len(row_labels)*0.5 + 2)))
        
        # Create heatmap
        sns.heatmap(data, annot=annotate, fmt=".2f", cmap=cmap,
                   xticklabels=col_labels, yticklabels=row_labels, ax=ax)
        
        if title:
            ax.set_title(title)
            
        if xlabel:
            ax.set_xlabel(xlabel)
            
        if ylabel:
            ax.set_ylabel(ylabel)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_attractor_trajectory(self, states, reference_state=None, decay=0.95,
                                     title=None, cmap=None, ref_cmap=None, 
                                     vmin=-1, vmax=1, save_path=None):
        """
        Visualize attractor trajectory as a weighted overlay.
        
        Args:
            states: List of 2D arrays representing states over time
            reference_state: Optional reference state to compare against
            decay: Factor to decrease intensity of older states
            title: Optional title
            cmap: Colormap for trajectory
            ref_cmap: Colormap for reference
            vmin, vmax: Value limits for colormap
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(15, 8))
        
        # Set default colormaps if not provided
        cmap = cmap or self.cmap
        ref_cmap = ref_cmap or FALSE_MEMORY_CMAP
        
        # Create layout
        if reference_state is not None:
            gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.05])
            ax_ref = plt.subplot(gs[0, 0])
            ax_final = plt.subplot(gs[0, 1])
            ax_overlay = plt.subplot(gs[0, 2])
            ax_colorbar = plt.subplot(gs[1, :])
        else:
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.05])
            ax_final = plt.subplot(gs[0, 0])
            ax_overlay = plt.subplot(gs[0, 1])
            ax_colorbar = plt.subplot(gs[1, :])
        
        # Display reference state if provided
        if reference_state is not None:
            im_ref = ax_ref.imshow(reference_state, cmap=ref_cmap, vmin=vmin, vmax=vmax)
            ax_ref.set_title("Reference State")
            ax_ref.set_axis_off()
        
        # Display final state
        final_state = states[-1]
        im_final = ax_final.imshow(final_state, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_final.set_title("Final State")
        ax_final.set_axis_off()
        
        # Create trajectory overlay
        overlay = np.zeros_like(final_state)
        weights = np.zeros_like(final_state)
        
        for i, state in enumerate(states):
            # Weight decreases exponentially for older states
            weight = decay ** (len(states) - i - 1)
            overlay += state * weight
            weights += weight
            
        # Normalize by total weight
        overlay = np.divide(overlay, weights, out=np.zeros_like(overlay), where=weights!=0)
        
        # Display overlay
        im_overlay = ax_overlay.imshow(overlay, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_overlay.set_title("Trajectory Overlay")
        ax_overlay.set_axis_off()
        
        # Add colorbar
        plt.colorbar(im_final, cax=ax_colorbar, orientation='horizontal')
        
        if title:
            fig.suptitle(title, fontsize=16)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_echo_effect(self, original_state, echo_states, reinforced_state,
                            title=None, vmin=-1, vmax=1, save_path=None):
        """
        Visualize echo effect showing original, echo, and reinforced states.
        
        Args:
            original_state: Initial state
            echo_states: States used in echo reinforcement
            reinforced_state: Final reinforced state
            title: Optional title
            vmin, vmax: Value limits for colormap
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_echo = len(echo_states)
        fig = plt.figure(figsize=(12, 8))
        
        # Create layout
        gs = gridspec.GridSpec(3, n_echo+1, height_ratios=[1, 1, 1])
        
        # First row: original state
        ax_original = plt.subplot(gs[0, :])
        im_original = ax_original.imshow(original_state, cmap=TRUE_MEMORY_CMAP, vmin=vmin, vmax=vmax)
        ax_original.set_title("Original State")
        ax_original.set_axis_off()
        
        # Second row: echo states
        for i, echo in enumerate(echo_states):
            ax_echo = plt.subplot(gs[1, i])
            im_echo = ax_echo.imshow(echo, cmap=FALSE_MEMORY_CMAP, vmin=vmin, vmax=vmax)
            ax_echo.set_title(f"Echo {i+1}")
            ax_echo.set_axis_off()
            
        # Add an arrow or explanation
        ax_arrow = plt.subplot(gs[1, -1])
        ax_arrow.text(0.5, 0.5, "→ Feedback →", ha='center', va='center', fontsize=12)
        ax_arrow.set_axis_off()
        
        # Third row: reinforced state
        ax_reinforced = plt.subplot(gs[2, :])
        im_reinforced = ax_reinforced.imshow(reinforced_state, cmap=TRUE_MEMORY_CMAP, vmin=vmin, vmax=vmax)
        ax_reinforced.set_title("Reinforced State")
        ax_reinforced.set_axis_off()
        
        # Add a colorbar
        fig.colorbar(im_original, ax=[ax_original, ax_reinforced], shrink=0.8)
        
        if title:
            fig.suptitle(title, fontsize=16)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_identity_metrics(self, identity_traces, metrics_list, title=None,
                                 save_path=None):
        """
        Visualize identity metrics across iterations.
        
        Args:
            identity_traces: List of IdentityTrace objects or similar
            metrics_list: List of metric names to visualize
            title: Optional title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not identity_traces or not metrics_list:
            return None
            
        # Number of traces and metrics
        n_traces = len(identity_traces)
        n_metrics = len(metrics_list)
        
        # Create figure with metrics as rows and traces as columns
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]
            
        # Plot each metric
        for i, metric_name in enumerate(metrics_list):
            ax = axes[i]
            
            for j, trace in enumerate(identity_traces):
                # Extract metric from trace (assuming trace has attribute or dictionary access)
                if hasattr(trace, metric_name):
                    metric_data = getattr(trace, metric_name)
                elif hasattr(trace, 'get') and callable(trace.get):
                    metric_data = trace.get(metric_name, [])
                else:
                    # Try dictionary-like access
                    try:
                        metric_data = trace[metric_name]
                    except (KeyError, TypeError):
                        continue
                
                # Plot metric data
                ax.plot(metric_data, 'o-', label=f"Trace {j+1}")
                
            ax.set_title(metric_name)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
        # Set x-axis label on bottom subplot
        axes[-1].set_xlabel('Iteration')
        
        if title:
            fig.suptitle(title, fontsize=16)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_adjacency_matrix(self, matrix, labels=None, title=None, 
                                 cmap="viridis", annotate=True, save_path=None):
        """
        Visualize adjacency matrix showing relationships between patterns.
        
        Args:
            matrix: 2D array of adjacency values
            labels: Optional labels for axes
            title: Optional title
            cmap: Colormap
            annotate: Whether to show values in cells
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if labels is None:
            labels = [f"Pattern {i+1}" for i in range(len(matrix))]
            
        fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.8 + 2), 
                                      max(6, len(labels)*0.8 + 2)))
        
        # Create heatmap
        sns.heatmap(matrix, annot=annotate, fmt=".2f", cmap=cmap,
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        if title:
            ax.set_title(title)
            
        ax.set_xlabel("Target Pattern")
        ax.set_ylabel("Source Pattern")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def visualize_failure_modes(self, modes, metrics, mode_names=None, 
                             title=None, save_path=None):
        """
        Visualize failure modes classification.
        
        Args:
            modes: List of mode classifications
            metrics: Dictionary with metric arrays (correlation, coherence, etc.)
            mode_names: Optional mapping of mode codes to readable names
            title: Optional title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not modes or not metrics:
            return None
            
        # Prepare readable mode names if not provided
        if mode_names is None:
            mode_names = {
                0: "Stable Recovery",
                1: "Graceful Deformation",
                2: "Ghost Recovery",
                3: "Hard Overwrite"
            }
            
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot correlation vs coherence with modes as colors
        if 'correlation' in metrics and 'coherence' in metrics:
            ax = axes[0]
            
            # Convert modes to numeric if needed
            if isinstance(modes[0], str):
                unique_modes = list(set(modes))
                mode_indices = [unique_modes.index(m) for m in modes]
            else:
                mode_indices = modes
                
            # Create scatter plot with mode colors
            scatter = ax.scatter(
                metrics['correlation'], 
                metrics['coherence'],
                c=mode_indices,
                cmap='viridis',
                s=100,
                alpha=0.7
            )
            
            # Add legend
            legend_elements = []
            for mode_idx, mode_name in mode_names.items():
                if mode_idx in mode_indices:
                    color = plt.cm.viridis(mode_idx / max(1, max(mode_indices)))
                    legend_elements.append(
                        Patch(facecolor=color, alpha=0.7, label=mode_name)
                    )
                    
            ax.legend(handles=legend_elements, loc='best')
            
            ax.set_xlabel("Correlation")
            ax.set_ylabel("Coherence")
            ax.set_title("Failure Modes in Correlation-Coherence Space")
            ax.grid(True, alpha=0.3)
            
            # Add threshold lines if appropriate
            if max(metrics['correlation']) > 0.5:
                ax.axvline(x=0.4, color='k', linestyle='--', alpha=0.3)
            
            if max(metrics['coherence']) > 0.15:
                ax.axhline(y=0.08, color='k', linestyle='--', alpha=0.3)
            
        # Plot mode count distribution
        ax = axes[1]
        mode_counts = {}
        
        # Count occurrences of each mode
        for mode in modes:
            mode_str = mode_names.get(mode, str(mode)) if isinstance(mode, int) else mode
            mode_counts[mode_str] = mode_counts.get(mode_str, 0) + 1
            
        # Create bar chart
        bars = ax.bar(list(mode_counts.keys()), list(mode_counts.values()))
        
        # Color bars according to mode (assuming similar order)
        for i, bar in enumerate(bars):
            if i < len(plt.cm.viridis.colors):
                bar.set_color(plt.cm.viridis(i / len(bars)))
                
        ax.set_xlabel("Failure Mode")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Failure Modes")
        ax.grid(axis='y', alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=16)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig