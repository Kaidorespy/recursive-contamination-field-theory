#!/usr/bin/env python3
# interference_analysis.py - Comprehensive analysis of RCFT interference data
# Usage: python interference_analysis.py [path_to_results_folder]

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import linregress

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis", 10)  # 10 colors for 10 counterfactuals

# Create output folders
def setup_output_dirs(base_dir="interference_analysis"):
    """Create output directories for analysis results"""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "clustering"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "stability"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "taxonomy"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "summary"), exist_ok=True)
    return base_dir

# Load and preprocess data
def load_data(results_dir):
    """Load and combine data from results files"""
    print(f"Loading data from {results_dir}...")
    
    # Try to find CSV files first (direct dataframes)
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if csv_files:
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            # If trial isn't in the data, extract it from filename
            if 'trial' not in df.columns and 'trial' in os.path.basename(file):
                trial_num = int(os.path.basename(file).split('_')[-1].split('.')[0])
                df['trial'] = trial_num
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    # If no CSVs, look for JSON files
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if json_files:
        data_list = []
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                
            # JSON structure can vary, so handle different formats
            if isinstance(data, list):
                for item in data:
                    # Flatten any nested structure
                    if isinstance(item, dict):
                        data_list.append(item)
            elif isinstance(data, dict):
                # Extract trial information from filename if possible
                if 'trial' not in data and 'trial' in os.path.basename(file):
                    trial_num = int(os.path.basename(file).split('_')[-1].split('.')[0])
                    data['trial'] = trial_num
                data_list.append(data)
                
        return pd.DataFrame(data_list)
    
    # If neither CSV nor JSON, try to find metrics files
    metrics_files = glob.glob(os.path.join(results_dir, "**/metrics.csv"), recursive=True)
    if metrics_files:
        dfs = []
        for file in metrics_files:
            # Extract CF and trial info from path
            path_parts = file.split(os.sep)
            cf_info = next((p for p in path_parts if p.startswith('cf_')), 'unknown')
            trial_info = next((p for p in path_parts if p.startswith('trial')), 'unknown')
            
            # Extract numbers
            cf_num = int(cf_info.split('_')[1]) if cf_info != 'unknown' else -1
            trial_num = int(trial_info.split('_')[1]) if trial_info != 'unknown' else -1
            
            df = pd.read_csv(file)
            df['counterfactual'] = f'cf_{cf_num}'
            df['trial'] = trial_num
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    # If all else fails
    print(f"No valid data files found in {results_dir}")
    return None

# Ensure the data has all the necessary columns
def preprocess_data(df):
    """Ensure data has all required columns and correct formatting"""
    # Check for required columns
    required_columns = ['counterfactual', 'trial', 'step']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Rename columns if needed (handle different naming conventions)
    column_mapping = {
        'cf': 'counterfactual',
        'recovery_corr': 'recovery_correlation',
        'time_step': 'step',
        'coherence_correlation_divergence': 'ccdi'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Calculate CCDI if not present but coherence and correlation are
    if 'ccdi' not in df.columns and 'coherence' in df.columns and 'correlation' in df.columns:
        df['ccdi'] = df['coherence'] - df['correlation']
    
    # Ensure counterfactual is formatted correctly
    if not all(str(cf).startswith('cf_') for cf in df['counterfactual'].unique()):
        df['counterfactual'] = df['counterfactual'].apply(lambda x: f'cf_{x}' if not str(x).startswith('cf_') else x)
    
    # Calculate recovery drop if not present
    if 'recovery_drop' not in df.columns and 'recovery_correlation' in df.columns:
        # Group by counterfactual and trial
        for cf in df['counterfactual'].unique():
            for trial in df['trial'].unique():
                mask = (df['counterfactual'] == cf) & (df['trial'] == trial)
                if not any(mask):
                    continue
                
                # Get initial correlation for this CF and trial
                initial_steps = df.loc[mask, 'step'].min()
                initial_corr = df.loc[(df['counterfactual'] == cf) & 
                                     (df['trial'] == trial) & 
                                     (df['step'] == initial_steps), 'recovery_correlation'].values
                
                if len(initial_corr) > 0:
                    initial_corr = initial_corr[0]
                    df.loc[mask, 'recovery_drop'] = initial_corr - df.loc[mask, 'recovery_correlation']
    
    return df

# Analysis Functions
def analyze_trajectories(df, output_dir, n_clusters=4):
    """Cluster recovery trajectories to find interference fingerprints"""
    print("Analyzing recovery trajectories...")
    
    # Extract unique counterfactuals and steps
    counterfactuals = sorted(df['counterfactual'].unique())
    
    # Create a pivot table with steps as columns and (cf, trial) as rows
    # This normalizes trajectory lengths for comparison
    trajectory_df = df.pivot_table(
        index=['counterfactual', 'trial'],
        columns='step',
        values='recovery_correlation'
    )
    
    # Handle missing values (in case different trials have different step counts)
    trajectory_df = trajectory_df.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    
    # Extract trajectory data for clustering
    X = trajectory_df.values
    X = np.nan_to_num(X)  # Replace any remaining NaNs
    
    # Find optimal number of clusters
    max_clusters = min(8, len(counterfactuals))
    silhouette_scores = []
    
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        score = silhouette_score(X, clusters)
        silhouette_scores.append((k, score))
    
    # Select number of clusters with highest silhouette score
    n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    print(f"Optimal number of clusters: {n_clusters}")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Add cluster assignments back to the dataframe
    cluster_df = pd.DataFrame({
        'counterfactual': [idx[0] for idx in trajectory_df.index],
        'trial': [idx[1] for idx in trajectory_df.index],
        'cluster': clusters
    })
    
    # Merge cluster information into original dataframe
    df = pd.merge(df, cluster_df, on=['counterfactual', 'trial'])
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame({
        'counterfactual': [idx[0] for idx in trajectory_df.index],
        'trial': [idx[1] for idx in trajectory_df.index],
        'cluster': clusters,
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1]
    })
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Plot points by counterfactual
    plt.subplot(1, 2, 1)
    for cf in counterfactuals:
        mask = pca_df['counterfactual'] == cf
        plt.scatter(pca_df.loc[mask, 'PCA1'], pca_df.loc[mask, 'PCA2'], 
                  label=cf, alpha=0.7, s=100)
    
    plt.title('Interference Fingerprints by Counterfactual')
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot points by cluster
    plt.subplot(1, 2, 2)
    for cluster_id in range(n_clusters):
        mask = pca_df['cluster'] == cluster_id
        plt.scatter(pca_df.loc[mask, 'PCA1'], pca_df.loc[mask, 'PCA2'], 
                  label=f'Cluster {cluster_id}', alpha=0.7, s=100)
    
    plt.title('Interference Fingerprints by Cluster')
    plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustering", "interference_fingerprints.png"), dpi=300)
    plt.close()
    
    # Plot cluster prototype curves
    plt.figure(figsize=(16, 12))
    
    # Plot average curves for each cluster
    for cluster_id in range(n_clusters):
        cluster_indices = pca_df.index[pca_df['cluster'] == cluster_id]
        cluster_curves = X[cluster_indices]
        mean_curve = np.mean(cluster_curves, axis=0)
        std_curve = np.std(cluster_curves, axis=0)
        
        plt.subplot(2, 2, cluster_id+1)
        steps = trajectory_df.columns
        
        plt.plot(steps, mean_curve, label=f'Cluster {cluster_id} Mean', linewidth=3)
        plt.fill_between(steps, mean_curve-std_curve, mean_curve+std_curve, alpha=0.3)
        
        # Add counterfactual distribution in this cluster
        cf_counts = pca_df[pca_df['cluster'] == cluster_id]['counterfactual'].value_counts()
        cf_list = ", ".join([f"{cf}({count})" for cf, count in cf_counts.items()])
        
        plt.title(f'Cluster {cluster_id} Prototype\nCounterfactuals: {cf_list}')
        plt.xlabel('Time Step')
        plt.ylabel('Recovery Correlation')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustering", "cluster_prototypes.png"), dpi=300)
    plt.close()
    
    # Create counterfactual vs cluster heatmap
    heatmap_data = pd.crosstab(pca_df['counterfactual'], pca_df['cluster'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='d')
    plt.title('Counterfactual Membership by Cluster')
    plt.ylabel('Counterfactual')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustering", "cf_cluster_heatmap.png"), dpi=300)
    plt.close()
    
    # Return enriched dataframe with cluster information
    return df

def analyze_stability(df, output_dir):
    """Analyze stability and variance metrics across counterfactuals"""
    print("Analyzing stability metrics...")
    
    # Extract unique counterfactuals
    counterfactuals = sorted(df['counterfactual'].unique())
    
    # Calculate stability metrics for each counterfactual
    stability_metrics = {}
    
    for cf in counterfactuals:
        cf_data = df[df['counterfactual'] == cf]
        
        # Get final step for this counterfactual
        final_step = cf_data['step'].max()
        
        # Compute variance of key metrics across trials
        recovery_var = cf_data.groupby(['trial', 'step'])['recovery_correlation'].mean().groupby('step').var().mean()
        
        # Calculate variance in CCDI if available
        if 'ccdi' in cf_data.columns:
            ccdi_var = cf_data.groupby(['trial', 'step'])['ccdi'].mean().groupby('step').var().mean()
        else:
            ccdi_var = np.nan
        
        # Calculate correlation drop if available
        if 'recovery_drop' in cf_data.columns:
            drop_var = cf_data.groupby(['trial', 'step'])['recovery_drop'].mean().groupby('step').var().mean()
        else:
            drop_var = np.nan
        
        # Final recovery metrics
        final_data = cf_data[cf_data['step'] == final_step]
        final_recovery = final_data['recovery_correlation']
        final_mean = final_recovery.mean()
        final_var = final_recovery.var()
        
        # CCDI at final step
        if 'ccdi' in final_data.columns:
            final_ccdi = final_data['ccdi'].mean()
        else:
            final_ccdi = np.nan
        
        # Recovery drop at final step
        if 'recovery_drop' in final_data.columns:
            final_drop = final_data['recovery_drop'].mean()
        else:
            final_drop = np.nan
        
        # Compute stability score - higher means stable good recovery
        stability_score = final_mean / (final_var + 0.01)
        
        # Store metrics
        stability_metrics[cf] = {
            'trajectory_variance': recovery_var,
            'ccdi_variance': ccdi_var,
            'drop_variance': drop_var,
            'final_recovery_mean': final_mean,
            'final_recovery_variance': final_var,
            'final_ccdi': final_ccdi,
            'final_drop': final_drop,
            'stability_score': stability_score
        }
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
    metrics_df.index.name = 'counterfactual'
    metrics_df.reset_index(inplace=True)
    
    # Plot stability vs severity
    plt.figure(figsize=(12, 10))
    
    # Create bubble chart
    plt.scatter(
        x=metrics_df['final_recovery_mean'],
        y=metrics_df['trajectory_variance'],
        s=metrics_df['stability_score'] * 100,  # Size based on stability score
        alpha=0.7,
        c=range(len(metrics_df)),  # Color by counterfactual
        cmap='viridis'
    )
    
    # Add labels
    for i, row in metrics_df.iterrows():
        plt.annotate(
            row['counterfactual'],
            (row['final_recovery_mean'], row['trajectory_variance']),
            fontsize=10
        )
    
    # Add quadrant lines at median values
    plt.axhline(y=metrics_df['trajectory_variance'].median(), 
               color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=metrics_df['final_recovery_mean'].median(), 
               color='r', linestyle='--', alpha=0.3)
    
    # Add quadrant labels
    x_med = metrics_df['final_recovery_mean'].median()
    y_med = metrics_df['trajectory_variance'].median()
    plt.text(x_med - 0.2, y_med + 0.02, "Unstable\nCatastrophic", ha='center', va='bottom', alpha=0.7)
    plt.text(x_med + 0.2, y_med + 0.02, "Unstable\nRecovery", ha='center', va='bottom', alpha=0.7)
    plt.text(x_med - 0.2, y_med - 0.02, "Stable\nCatastrophic", ha='center', va='top', alpha=0.7)
    plt.text(x_med + 0.2, y_med - 0.02, "Stable\nRecovery", ha='center', va='top', alpha=0.7)
    
    plt.title('Counterfactual Stability vs Severity')
    plt.xlabel('Final Recovery (higher = better)')
    plt.ylabel('Trajectory Variance (higher = unstable)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability", "stability_vs_severity.png"), dpi=300)
    plt.close()
    
    # Create 3D bubble chart with CCDI
    if not metrics_df['final_ccdi'].isna().all():
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            metrics_df['final_recovery_mean'],
            metrics_df['final_drop'],
            metrics_df['final_ccdi'],
            s=metrics_df['stability_score'] * 100,
            c=range(len(metrics_df)),
            cmap='viridis',
            alpha=0.7
        )
        
        # Add labels
        for i, row in metrics_df.iterrows():
            ax.text(
                row['final_recovery_mean'],
                row['final_drop'],
                row['final_ccdi'],
                row['counterfactual'],
                fontsize=9
            )
        
        ax.set_xlabel('Final Recovery')
        ax.set_ylabel('Recovery Drop')
        ax.set_zlabel('CCDI')
        ax.set_title('3D Stability Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "stability", "3d_stability.png"), dpi=300)
        plt.close()
    
    # Create correlation matrix of metrics
    corr_matrix = metrics_df.drop('counterfactual', axis=1).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Stability Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability", "metrics_correlation.png"), dpi=300)
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, "stability", "stability_metrics.csv"), index=False)
    
    return metrics_df

def analyze_trajectories_over_time(df, output_dir):
    """Create mean/variance curves for key metrics over time"""
    print("Analyzing recovery trajectories over time...")
    
    # Extract unique counterfactuals
    counterfactuals = sorted(df['counterfactual'].unique())
    
    # Create averaged recovery curves with variance bands
    plt.figure(figsize=(15, 18))
    
    # Plot for Recovery Correlation
    plt.subplot(3, 1, 1)
    for cf in counterfactuals:
        cf_data = df[df['counterfactual'] == cf]
        
        # Group by step and calculate mean and std
        stats = cf_data.groupby('step')['recovery_correlation'].agg(['mean', 'std']).reset_index()
        steps = stats['step']
        mean_curve = stats['mean']
        std_curve = stats['std']
        
        # Plot mean with std band
        plt.plot(steps, mean_curve, label=cf)
        plt.fill_between(steps, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)
    
    plt.title('Recovery Correlation by Counterfactual')
    plt.xlabel('Time Step')
    plt.ylabel('Recovery Correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot for Recovery Drop
    if 'recovery_drop' in df.columns:
        plt.subplot(3, 1, 2)
        for cf in counterfactuals:
            cf_data = df[df['counterfactual'] == cf]
            
            # Group by step and calculate mean and std
            stats = cf_data.groupby('step')['recovery_drop'].agg(['mean', 'std']).reset_index()
            steps = stats['step']
            mean_curve = stats['mean']
            std_curve = stats['std']
            
            # Plot mean with std band
            plt.plot(steps, mean_curve, label=cf)
            plt.fill_between(steps, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)
        
        plt.title('Recovery Drop by Counterfactual')
        plt.xlabel('Time Step')
        plt.ylabel('Recovery Drop')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    
    # Plot for CCDI
    if 'ccdi' in df.columns:
        plt.subplot(3, 1, 3)
        for cf in counterfactuals:
            cf_data = df[df['counterfactual'] == cf]
            
            # Group by step and calculate mean and std
            stats = cf_data.groupby('step')['ccdi'].agg(['mean', 'std']).reset_index()
            steps = stats['step']
            mean_curve = stats['mean']
            std_curve = stats['std']
            
            # Plot mean with std band
            plt.plot(steps, mean_curve, label=cf)
            plt.fill_between(steps, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)
        
        plt.title('CCDI by Counterfactual')
        plt.xlabel('Time Step')
        plt.ylabel('CCDI')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectories", "recovery_trajectories.png"), dpi=300)
    plt.close()
    
    # Create individual trajectory plots for each counterfactual
    for cf in counterfactuals:
        cf_data = df[df['counterfactual'] == cf]
        
        plt.figure(figsize=(14, 10))
        
        # Get unique trials
        trials = sorted(cf_data['trial'].unique())
        
        # Plot recovery correlation for each trial
        for trial in trials:
            trial_data = cf_data[cf_data['trial'] == trial]
            plt.plot(trial_data['step'], trial_data['recovery_correlation'], 
                    label=f'Trial {trial}', alpha=0.7)
        
        # Calculate and plot mean across trials
        mean_curve = cf_data.groupby('step')['recovery_correlation'].mean()
        plt.plot(mean_curve.index, mean_curve.values, 'k--', 
                label='Mean', linewidth=2)
        
        plt.title(f'Recovery Correlation for {cf}')
        plt.xlabel('Time Step')
        plt.ylabel('Recovery Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trajectories", f"{cf}_recovery.png"), dpi=300)
        plt.close()
    
    # Create correlation vs CCDI trajectory plot if CCDI is available
    if 'ccdi' in df.columns and 'recovery_correlation' in df.columns:
        plt.figure(figsize=(14, 10))
        
        for cf in counterfactuals:
            cf_data = df[df['counterfactual'] == cf]
            
            # Calculate means across trials
            means = cf_data.groupby('step').agg({
                'recovery_correlation': 'mean',
                'ccdi': 'mean'
            }).reset_index()
            
            # Plot as a trajectory in CCDI vs Correlation space
            plt.plot(means['recovery_correlation'], means['ccdi'], 'o-', 
                    label=cf, linewidth=2, markersize=5)
            
            # Mark start and end points
            plt.plot(means.iloc[0]['recovery_correlation'], means.iloc[0]['ccdi'], 
                    'o', color='green', markersize=8)
            plt.plot(means.iloc[-1]['recovery_correlation'], means.iloc[-1]['ccdi'], 
                    's', color='red', markersize=8)
        
        plt.title('Recovery Trajectories in Correlation-CCDI Space')
        plt.xlabel('Recovery Correlation')
        plt.ylabel('CCDI')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add threshold lines for failure mode taxonomy
        plt.axhline(y=0.1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0.4, color='k', linestyle='--', alpha=0.3)
        
        # Add zone labels
        plt.text(0.7, 0.15, "Graceful Deformation", ha='center', fontsize=12)
        plt.text(0.2, 0.05, "Hard Overwrite", ha='center', fontsize=12)
        plt.text(0.2, 0.15, "Ghost Recovery", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "trajectories", "corr_ccdi_space.png"), dpi=300)
        plt.close()

def classify_failure_modes(df, metrics_df, output_dir):
    """Classify counterfactuals into failure mode taxonomy"""
    print("Classifying failure modes...")
    
    # Define failure mode thresholds
    # These can be adjusted based on your specific data
    # For illustration, we'll use the following thresholds:
    ccdi_threshold = 0.08  # Above this is high CCDI
    recovery_threshold = 0.4  # Below this is low recovery
    drop_threshold = 0.3  # Above this is significant drop
    
    # Create classification function
    def classify_mode(row):
        final_recovery = row['final_recovery_mean']
        final_ccdi = row['final_ccdi']
        final_drop = row['final_drop']
        
        # Handle missing values
        if pd.isna(final_ccdi) or pd.isna(final_drop):
            return "Unknown"
        
        if final_ccdi > ccdi_threshold and final_recovery >= recovery_threshold:
            return "Graceful Deformation"
        elif final_ccdi > ccdi_threshold and final_recovery < recovery_threshold:
            return "Ghost Recovery"
        elif final_recovery < recovery_threshold:
            return "Hard Overwrite"
        else:
            return "Stable Recovery"
    
    # Apply classification and add to metrics_df
    metrics_df['failure_mode'] = metrics_df.apply(classify_mode, axis=1)
    
    # Create summary plot
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot of final recovery vs CCDI, colored by failure mode
    scatter = plt.scatter(
        metrics_df['final_recovery_mean'],
        metrics_df['final_ccdi'],
        s=100,
        c=pd.Categorical(metrics_df['failure_mode']).codes,
        cmap='viridis',
        alpha=0.7
    )
    
    # Add labels
    for i, row in metrics_df.iterrows():
        plt.annotate(
            row['counterfactual'],
            (row['final_recovery_mean'], row['final_ccdi']),
            fontsize=10
        )
    
    # Add threshold lines
    plt.axhline(y=ccdi_threshold, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=recovery_threshold, color='k', linestyle='--', alpha=0.3)
    
    # Add zone labels
    plt.text(0.7, ccdi_threshold+0.02, "Graceful Deformation", ha='center', fontsize=12)
    plt.text(recovery_threshold-0.15, 0.05, "Hard Overwrite", ha='center', fontsize=12)
    plt.text(recovery_threshold-0.15, ccdi_threshold+0.02, "Ghost Recovery", ha='center', fontsize=12)
    plt.text(0.7, 0.05, "Stable Recovery", ha='center', fontsize=12)
    
    # Add legend
    legend1 = plt.legend(scatter.legend_elements()[0], 
                         metrics_df['failure_mode'].unique(),
                         title="Failure Mode",
                         loc="upper left")
    plt.gca().add_artist(legend1)
    
    plt.title('Failure Mode Taxonomy')
    plt.xlabel('Final Recovery Correlation')
    plt.ylabel('Final CCDI')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "taxonomy", "failure_modes.png"), dpi=300)
    plt.close()
    
    # Create bar chart of failure mode counts
    mode_counts = metrics_df['failure_mode'].value_counts()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(mode_counts.index, mode_counts.values, color=plt.cm.viridis(range(len(mode_counts))))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.title('Failure Mode Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "taxonomy", "failure_mode_counts.png"), dpi=300)
    plt.close()
    
    # Create heatmap of failure modes vs counterfactuals
    mode_cf_matrix = pd.crosstab(metrics_df['failure_mode'], metrics_df['counterfactual'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(mode_cf_matrix, annot=True, cmap='viridis', fmt='d')
    plt.title('Failure Modes by Counterfactual')
    plt.xlabel('Counterfactual')
    plt.ylabel('Failure Mode')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "taxonomy", "mode_cf_heatmap.png"), dpi=300)
    plt.close()
    
    # Save failure mode classification to CSV
    metrics_df.to_csv(os.path.join(output_dir, "taxonomy", "failure_mode_classification.csv"), index=False)
    
    return metrics_df

def create_summary_dashboard(df, metrics_df, output_dir):
    """Create a comprehensive summary dashboard"""
    print("Creating summary dashboard...")
    
    # Create a single comprehensive summary figure
    plt.figure(figsize=(20, 16))
    
    # 1. Failure Mode Classification
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(
        metrics_df['final_recovery_mean'],
        metrics_df['final_ccdi'],
        s=100,
        c=pd.Categorical(metrics_df['failure_mode']).codes,
        cmap='viridis',
        alpha=0.7
    )
    
    for i, row in metrics_df.iterrows():
        plt.annotate(
            row['counterfactual'],
            (row['final_recovery_mean'], row['final_ccdi']),
            fontsize=10
        )
    
    # Add threshold lines
    ccdi_threshold = 0.08
    recovery_threshold = 0.4
    plt.axhline(y=ccdi_threshold, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=recovery_threshold, color='k', linestyle='--', alpha=0.3)
    
    legend1 = plt.legend(scatter.legend_elements()[0], 
                       metrics_df['failure_mode'].unique(),
                       title="Failure Mode",
                       loc="upper left")
    plt.gca().add_artist(legend1)
    
    plt.title('Failure Mode Taxonomy')
    plt.xlabel('Final Recovery Correlation')
    plt.ylabel('Final CCDI')
    plt.grid(True, alpha=0.3)
    
    # 2. Recovery Trajectories
    plt.subplot(2, 2, 2)
    counterfactuals = sorted(df['counterfactual'].unique())
    
    for cf in counterfactuals:
        cf_data = df[df['counterfactual'] == cf]
        
        # Group by step and calculate mean
        mean_curve = cf_data.groupby('step')['recovery_correlation'].mean()
        steps = mean_curve.index
        
        # Get failure mode color
        mode = metrics_df[metrics_df['counterfactual'] == cf]['failure_mode'].values[0]
        mode_idx = list(metrics_df['failure_mode'].unique()).index(mode)
        color = plt.cm.viridis(mode_idx / len(metrics_df['failure_mode'].unique()))
        
        # Plot with mode-based color
        plt.plot(steps, mean_curve, label=f"{cf} ({mode})", color=color)
    
    plt.title('Recovery Trajectories by Counterfactual')
    plt.xlabel('Time Step')
    plt.ylabel('Recovery Correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 3. Stability Analysis
    plt.subplot(2, 2, 3)
    
    scatter = plt.scatter(
        metrics_df['final_recovery_mean'],
        metrics_df['trajectory_variance'],
        s=metrics_df['stability_score'] * 100,
        c=pd.Categorical(metrics_df['failure_mode']).codes,
        cmap='viridis',
        alpha=0.7
    )
    
    for i, row in metrics_df.iterrows():
        plt.annotate(
            row['counterfactual'],
            (row['final_recovery_mean'], row['trajectory_variance']),
            fontsize=10
        )
    
    plt.axhline(y=metrics_df['trajectory_variance'].median(), 
               color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=metrics_df['final_recovery_mean'].median(), 
               color='r', linestyle='--', alpha=0.3)
    
    plt.title('Stability vs Recovery Quality')
    plt.xlabel('Final Recovery (higher = better)')
    plt.ylabel('Trajectory Variance (higher = unstable)')
    plt.grid(True, alpha=0.3)
    
    # 4. Metric Correlations
    plt.subplot(2, 2, 4)
    
    # Select key metrics for correlation
    key_metrics = ['final_recovery_mean', 'final_ccdi', 'trajectory_variance', 'stability_score']
    key_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    corr_matrix = metrics_df[key_metrics].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('Correlation Between Key Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary", "interference_summary.png"), dpi=300)
    plt.close()
    
    # Create individual summary for each counterfactual
    for cf in counterfactuals:
        cf_data = df[df['counterfactual'] == cf]
        cf_metrics = metrics_df[metrics_df['counterfactual'] == cf]
        
        plt.figure(figsize=(15, 10))
        
        # Upper left: Recovery correlation over time
        plt.subplot(2, 2, 1)
        
        # Plot individual trials
        trials = sorted(cf_data['trial'].unique())
        for trial in trials:
            trial_data = cf_data[cf_data['trial'] == trial]
            plt.plot(trial_data['step'], trial_data['recovery_correlation'], 
                    alpha=0.3, label=f'Trial {trial}' if trial == trials[0] else None)
        
        # Plot mean
        mean_curve = cf_data.groupby('step')['recovery_correlation'].mean()
        plt.plot(mean_curve.index, mean_curve.values, 'k-', linewidth=2, label='Mean')
        
        plt.title(f'Recovery Correlation for {cf}')
        plt.xlabel('Time Step')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Upper right: CCDI over time if available
        plt.subplot(2, 2, 2)
        
        if 'ccdi' in cf_data.columns:
            # Plot individual trials
            for trial in trials:
                trial_data = cf_data[cf_data['trial'] == trial]
                plt.plot(trial_data['step'], trial_data['ccdi'], 
                        alpha=0.3)
            
            # Plot mean
            mean_curve = cf_data.groupby('step')['ccdi'].mean()
            plt.plot(mean_curve.index, mean_curve.values, 'k-', linewidth=2)
            
            plt.title(f'CCDI for {cf}')
            plt.xlabel('Time Step')
            plt.ylabel('CCDI')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "CCDI data not available", 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Lower left: Final state metrics
        plt.subplot(2, 2, 3)
        
        metrics_to_plot = ['final_recovery_mean', 'final_ccdi', 'stability_score']
        metrics_to_plot = [m for m in metrics_to_plot if m in cf_metrics.columns]
        
        if metrics_to_plot:
            values = cf_metrics[metrics_to_plot].values[0]
            
            barlist = plt.bar(metrics_to_plot, values)
            
            # Customize bar colors
            for i, metric in enumerate(metrics_to_plot):
                if metric == 'final_recovery_mean' and values[i] < 0.4:
                    barlist[i].set_color('r')
                elif metric == 'final_ccdi' and values[i] > 0.08:
                    barlist[i].set_color('orange')
            
            plt.title(f'Final Metrics for {cf}')
            plt.ylabel('Value')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, "Final metrics not available", 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        # Lower right: Failure mode classification
        plt.subplot(2, 2, 4)
        
        if 'failure_mode' in cf_metrics.columns:
            failure_mode = cf_metrics['failure_mode'].values[0]
            
            # Create a pie chart with just one section
            mode_colors = {
                'Graceful Deformation': 'lightblue',
                'Ghost Recovery': 'orange',
                'Hard Overwrite': 'red',
                'Stable Recovery': 'green',
                'Unknown': 'gray'
            }
            
            plt.pie([1], labels=[failure_mode], 
                   colors=[mode_colors.get(failure_mode, 'gray')],
                   autopct='%1.0f%%', startangle=90)
            
            plt.title(f'Failure Mode: {failure_mode}')
        else:
            plt.text(0.5, 0.5, "Failure mode classification not available", 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.suptitle(f'Summary for {cf}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary", f"{cf}_summary.png"), dpi=300)
        plt.close()

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to current directory if no argument is provided
        results_dir = "."
    
    # Setup output directories
    output_dir = setup_output_dirs()
    
    # Load data
    df = load_data(results_dir)
    
    if df is None or df.empty:
        print("No valid data found. Exiting.")
        return
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Run analyses
    df = analyze_trajectories(df, output_dir)
    metrics_df = analyze_stability(df, output_dir)
    analyze_trajectories_over_time(df, output_dir)
    metrics_df = classify_failure_modes(df, metrics_df, output_dir)
    create_summary_dashboard(df, metrics_df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
