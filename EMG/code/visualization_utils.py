"""
Visualization utilities for EMG classification results - Multi-Model Version
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd


def plot_confusion_matrix(cm: np.ndarray, class_names: Dict, 
                         title: str = 'Confusion Matrix',
                         figsize: tuple = (10, 8)) -> None:
    """Plot confusion matrix as heatmap"""
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create labels
    labels = [class_names[i] for i in sorted(class_names.keys())]
    
    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names: List[str] = None,
                           top_n: int = 20, figsize: tuple = (12, 8)) -> None:
    """Plot feature importance for tree-based models"""
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'estimators_'):
        # For ensemble models, try to get from first estimator
        if hasattr(model.estimators_[0], 'feature_importances_'):
            importances = model.estimators_[0].feature_importances_
        elif hasattr(model.estimators_[0][1], 'feature_importances_'):
            importances = model.estimators_[0][1].feature_importances_
        else:
            print("Model does not have feature importances")
            return
    else:
        print("Model does not have feature importances")
        return
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # Get top features
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]
    
    # Plot
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), top_importances[::-1])
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_distribution(labels: np.ndarray, class_names: Dict,
                           figsize: tuple = (10, 6)) -> None:
    """Plot class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    names = [class_names[i] for i in unique]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(names, counts, color=sns.color_palette('husl', len(unique)))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(model_results: List[Dict], figsize: tuple = (16, 10)) -> None:
    """
    Plot comprehensive model comparison
    
    Args:
        model_results: List of dictionaries with model results
        figsize: Figure size
    """
    # Sort by test accuracy or CV score
    has_test = model_results[0]['test_accuracy'] is not None
    if has_test:
        sorted_results = sorted(model_results, 
                               key=lambda x: x['test_accuracy'], 
                               reverse=True)
    else:
        sorted_results = sorted(model_results, 
                               key=lambda x: x['cv_mean'], 
                               reverse=True)
    
    model_names = [r['model_name'] for r in sorted_results]
    train_accs = [r['train_accuracy'] for r in sorted_results]
    cv_means = [r['cv_mean'] for r in sorted_results]
    cv_stds = [r['cv_std'] for r in sorted_results]
    train_times = [r['train_time'] for r in sorted_results]
    
    if has_test:
        test_accs = [r['test_accuracy'] for r in sorted_results]
    
    fig = plt.figure(figsize=figsize)
    
    # Create subplots
    if has_test:
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(2, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
    
    # Plot 1: Accuracy comparison
    x = np.arange(len(model_names))
    width = 0.35 if has_test else 0.25
    
    if has_test:
        ax1.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
        ax1.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    else:
        ax1.bar(x, train_accs, width, label='Train Accuracy', alpha=0.8)
    
    # Add CV scores with error bars
    ax1.errorbar(x, cv_means, yerr=cv_stds, fmt='o', color='red', 
                 markersize=8, capsize=5, label='CV Score', linewidth=2)
    
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.5, 1.05])
    
    # Add value labels on bars
    for i, (train_acc, cv_mean) in enumerate(zip(train_accs, cv_means)):
        ax1.text(i, train_acc + 0.01, f'{train_acc:.3f}', 
                ha='center', va='bottom', fontsize=8)
        if has_test:
            ax1.text(i, test_accs[i] + 0.01, f'{test_accs[i]:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Training time
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = ax2.barh(model_names, train_times, color=colors, alpha=0.8)
    ax2.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, train_times)):
        ax2.text(time, bar.get_y() + bar.get_height()/2, 
                f'{time:.2f}s', va='center', fontsize=9)
    
    if has_test:
        # Plot 3: Accuracy vs Training Time scatter
        scatter = ax3.scatter(train_times, test_accs, 
                            s=200, alpha=0.6, c=range(len(model_names)),
                            cmap='viridis', edgecolors='black', linewidth=1.5)
        
        # Annotate points
        for i, name in enumerate(model_names):
            ax3.annotate(name, (train_times[i], test_accs[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax3.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Accuracy vs Training Time', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add best model highlight
        best_idx = test_accs.index(max(test_accs))
        ax3.scatter(train_times[best_idx], test_accs[best_idx],
                   s=300, facecolors='none', edgecolors='red', 
                   linewidths=3, label='Best Model')
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Rank':<6} {'Model':<30} {'Train Acc':<12} {'Test Acc':<12} "
          f"{'CV Score':<15} {'Time (s)':<10}")
    print("-"*100)
    
    for i, result in enumerate(sorted_results, 1):
        test_acc_str = f"{result['test_accuracy']:.4f}" if result['test_accuracy'] is not None else "N/A"
        cv_str = f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}"
        print(f"{i:<6} {result['model_name']:<30} {result['train_accuracy']:<12.4f} "
              f"{test_acc_str:<12} {cv_str:<15} {result['train_time']:<10.2f}")
    
    print("="*100)


def plot_cv_detailed(cv_results: Dict[str, List[Dict]], class_names: Dict,
                    figsize: tuple = (16, 10)) -> None:
    """
    Plot detailed cross-validation results for multiple models
    
    Args:
        cv_results: Dictionary mapping model names to list of CV fold results
        class_names: Dictionary mapping class IDs to names
        figsize: Figure size
    """
    n_models = len(cv_results)
    model_names = list(cv_results.keys())
    n_folds = len(cv_results[model_names[0]])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Accuracy per fold for each model
    ax1 = axes[0, 0]
    for model_name in model_names:
        accuracies = [r['accuracy'] for r in cv_results[model_name]]
        trials = [r['test_trial'] for r in cv_results[model_name]]
        ax1.plot(trials, accuracies, marker='o', linewidth=2, 
                markersize=6, label=model_name, alpha=0.7)
    
    ax1.set_xlabel('Test Trial', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy per Fold', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Plot 2: Mean accuracy with error bars
    ax2 = axes[0, 1]
    means = []
    stds = []
    for model_name in model_names:
        accuracies = [r['accuracy'] for r in cv_results[model_name]]
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))
    
    x = np.arange(len(model_names))
    bars = ax2.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=sns.color_palette('husl', n_models))
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Mean CV Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.5, 1.0])
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(i, mean + std + 0.01, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Box plot of accuracies
    ax3 = axes[1, 0]
    data_to_plot = []
    for model_name in model_names:
        accuracies = [r['accuracy'] for r in cv_results[model_name]]
        data_to_plot.append(accuracies)
    
    bp = ax3.boxplot(data_to_plot, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette('husl', n_models)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0.5, 1.0])
    
    # Plot 4: Per-class F1 scores for best model
    ax4 = axes[1, 1]
    best_model = model_names[np.argmax(means)]
    
    # Average per-class metrics across folds
    n_classes = len(class_names)
    avg_f1_per_class = np.zeros(n_classes)
    
    for result in cv_results[best_model]:
        avg_f1_per_class += result['per_class_metrics']['f1']
    avg_f1_per_class /= n_folds
    
    class_labels = [class_names[i] for i in sorted(class_names.keys())]
    x_classes = np.arange(n_classes)
    bars = ax4.bar(x_classes, avg_f1_per_class, alpha=0.7,
                   color=sns.color_palette('husl', n_classes))
    
    ax4.set_ylabel('F1-Score', fontsize=11)
    ax4.set_title(f'Per-Class F1-Score ({best_model})', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_classes)
    ax4.set_xticklabels(class_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.0])
    
    # Add value labels
    for i, score in enumerate(avg_f1_per_class):
        ax4.text(i, score + 0.01, f'{score:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('cv_detailed_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_emg_signals(data: np.ndarray, channels: List[int] = None,
                    title: str = 'EMG Signals', 
                    sampling_rate: int = 500,
                    figsize: tuple = (14, 10)) -> None:
    """Plot raw EMG signals"""
    n_samples, n_channels = data.shape
    time = np.arange(n_samples) / sampling_rate
    
    if channels is None:
        channels = list(range(n_channels))
    
    n_plot = len(channels)
    fig, axes = plt.subplots(n_plot, 1, figsize=figsize, sharex=True)
    
    if n_plot == 1:
        axes = [axes]
    
    for i, ch in enumerate(channels):
        axes[i].plot(time, data[:, ch], linewidth=0.5)
        axes[i].set_ylabel(f'Channel {ch}\n(μV)', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(data[:, ch])
        std_val = np.std(data[:, ch])
        axes[i].text(0.02, 0.95, f'μ={mean_val:.1f}, σ={std_val:.1f}',
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('emg_signals.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_results_summary(cv_results: List[Dict], class_names: Dict,
                          output_file: str = 'results_summary.txt') -> None:
    """Create a text file with comprehensive results summary"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EMG MOVEMENT CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        accuracies = [r['accuracy'] for r in cv_results]
        precisions = [r['precision'] for r in cv_results]
        recalls = [r['recall'] for r in cv_results]
        f1_scores = [r['f1'] for r in cv_results]
        
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of folds: {len(cv_results)}\n\n")
        
        f.write(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n")
        f.write(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
        f.write(f"F1-Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n\n")
        
        # Per-fold results
        f.write("\nPER-FOLD RESULTS\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(cv_results, 1):
            f.write(f"\nFold {i} (Test Trial {result['test_trial']}):\n")
            f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall:    {result['recall']:.4f}\n")
            f.write(f"  F1-Score:  {result['f1']:.4f}\n")
        
        # Average confusion matrix
        f.write("\n\nAVERAGE CONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        avg_cm = np.mean([r['confusion_matrix'] for r in cv_results], axis=0)
        f.write(str(avg_cm) + "\n")
        
        # Class names
        f.write("\n\nCLASS MAPPING\n")
        f.write("-"*80 + "\n")
        for class_id, class_name in sorted(class_names.items()):
            f.write(f"Class {class_id}: {class_name}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Results summary saved to {output_file}")


if __name__ == "__main__":
    print("This module contains visualization utilities.")
    print("Import and use the functions in your main analysis script.")