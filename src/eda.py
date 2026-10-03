import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURE_NAMES = [
    'mean_ip', 'std_ip', 'kurt_ip', 'skew_ip',
    'mean_snr', 'std_snr', 'kurt_snr', 'skew_snr'
]
TARGET_NAME = 'target'

def load_data(filepath='data/HTRU_2.csv'):
    """Loads HTRU2 dataset with proper feature column names."""
    df = pd.read_csv(filepath, header=None)
    df.columns = FEATURE_NAMES + [TARGET_NAME]
    return df

def run_eda(df, output_dir='outputs/plots'):
    """Runs complete EDA and saves publication-quality plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Class imbalance analysis
    class_counts = df[TARGET_NAME].value_counts()
    print("=== Class Imbalance Analysis ===")
    print(f"Non-pulsars (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"Pulsars (1):     {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")
    print(f"Imbalance ratio: 1:{class_counts[0]/class_counts[1]:.1f}")

    # Plot 1: Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_NAME, data=df, palette=['#1f77b4', '#ff7f0e'])
    plt.title('HTRU2 Class Distribution (Imbalance)', fontsize=12, fontweight='bold')
    plt.xlabel('Class (0: Non-Pulsar, 1: Pulsar)')
    plt.ylabel('Count')
    for i, count in enumerate([class_counts[0], class_counts[1]]):
        plt.text(i, count + 200, f"{count}\n({count/len(df)*100:.1f}%)", ha='center', fontweight='bold')
    plt.ylim(0, 18500)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    plt.close()

    # Plot 2: Feature Distributions by Class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, col in enumerate(FEATURE_NAMES):
        sns.kdeplot(data=df[df[TARGET_NAME] == 0][col], ax=axes[idx], color='blue', label='Non-Pulsar', fill=True, alpha=0.3)
        sns.kdeplot(data=df[df[TARGET_NAME] == 1][col], ax=axes[idx], color='orange', label='Pulsar', fill=True, alpha=0.3)
        axes[idx].set_title(col, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].legend(loc='upper right', fontsize=8)
    plt.suptitle('Feature Probability Distributions by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300)
    plt.close()

    # Plot 3: Feature Correlation Matrix
    plt.figure(figsize=(8, 6))
    corr = df[FEATURE_NAMES].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    
    print(f"EDA plots saved to {output_dir}/")
    return class_counts

if __name__ == '__main__':
    df = load_data()
    run_eda(df)
