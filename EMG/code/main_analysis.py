"""
Main analysis script for EMG movement classification - Multi-Model Version
Complete pipeline with automatic model selection
"""

import numpy as np
from pathlib import Path
import sys
import pandas as pd

# Import custom modules
from emg_classification_pipeline import (
    EMGDataLoader,
    EMGClassifier,
    evaluate_model
)
from visualization_utils import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_class_distribution,
    plot_emg_signals,
    create_results_summary,
    plot_model_comparison,
    plot_cv_detailed
)


def run_complete_analysis(data_dir: str, n_trials: int = 49, n_reps: int = 5,
                         window_size_ms: int = 100, overlap: float = 0.5,
                         sampling_rate: int = 500,
                         include_ensembles: bool = True,
                         optimize_best: bool = True):
    """
    Run complete EMG classification analysis pipeline with model comparison
    
    Args:
        data_dir: Directory containing EMG data files
        n_trials: Number of trials
        n_reps: Number of repetitions per trial
        window_size_ms: Window size in milliseconds
        overlap: Window overlap ratio
        sampling_rate: Sampling rate in Hz
        include_ensembles: Whether to train ensemble models
        optimize_best: Whether to optimize the best model's hyperparameters
    """
    print("="*80)
    print("EMG MOVEMENT CLASSIFICATION - MULTI-MODEL ANALYSIS")
    print("="*80)
    
    # 1. Initialize components
    print("\n[1/7] Initializing components...")
    data_loader = EMGDataLoader(data_dir)
    classifier = EMGClassifier(
        window_size_ms=window_size_ms,
        overlap=overlap,
        sampling_rate=sampling_rate
    )
    
    # 2. Load data
    print("\n[2/7] Loading dataset...")
    data_list, labels, metadata = data_loader.load_dataset(n_trials, n_reps)
    print(f"Loaded {len(data_list)} files")
    print(f"Total samples: {sum(len(d) for d in data_list)}")
    
    # 3. Visualize raw data sample
    print("\n[3/7] Visualizing sample EMG signals...")
    sample_data = data_list[0][:500]  # First 1 second at 500 Hz
    plot_emg_signals(sample_data, title='Sample EMG Signals (1 second)', 
                     sampling_rate=sampling_rate)
    
    # 4. Prepare data
    print("\n[4/7] Preparing features...")
    X, y = classifier.prepare_data(data_list, labels)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Plot class distribution
    plot_class_distribution(y, data_loader.class_names)
    
    # 5. Split data for model comparison (use last trial as holdout test set)
    print("\n[5/7] Splitting data (Train: Trials 1-9, Test: Trial 10)...")
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    for data, label, meta in zip(data_list, labels, metadata):
        if meta['trial'] == n_trials:  # Last trial for testing
            test_data.append(data)
            test_labels.append(label)
        else:
            train_data.append(data)
            train_labels.append(label)
    
    X_train, y_train = classifier.prepare_data(train_data, train_labels)
    X_test, y_test = classifier.prepare_data(test_data, test_labels)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 6. Train and compare all models
    print("\n[6/7] Training and comparing multiple models...")
    print("This will take several minutes...")
    
    model_results = classifier.train_and_compare_models(
        X_train, y_train,
        X_test, y_test,
        include_ensembles=include_ensembles,
        cv_folds=5
    )
    
    # Visualize model comparison
    plot_model_comparison(model_results)
    
    # 7. Optimize best model (optional)
    if optimize_best:
        print("\n[7/7] Optimizing best model hyperparameters...")
        try:
            classifier.optimize_best_model(X_train, y_train, cv_folds=3)
            
            # Re-evaluate after optimization
            print("\nRe-evaluating optimized model...")
            optimized_results = evaluate_model(
                classifier, X_test, y_test, data_loader.class_names
            )
            
            # Plot confusion matrix for optimized model
            plot_confusion_matrix(
                optimized_results['confusion_matrix'],
                data_loader.class_names,
                title=f'Confusion Matrix - {classifier.model_name} (Optimized)'
            )
        except Exception as e:
            print(f"Could not optimize model: {e}")
            print("Continuing with best model from comparison...")
    else:
        print("\n[7/7] Skipping optimization (optimize_best=False)")
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("="*80)
    final_results = evaluate_model(classifier, X_test, y_test, data_loader.class_names)
    
    # Plot feature importance
    try:
        n_features_per_channel = 9
        feature_names = []
        for ch in range(8):
            feature_names.extend([
                f'CH{ch}_MAV', f'CH{ch}_RMS', f'CH{ch}_WL',
                f'CH{ch}_ZCR', f'CH{ch}_SSC', f'CH{ch}_STD',
                f'CH{ch}_VAR', f'CH{ch}_Kurt', f'CH{ch}_Skew'
            ])
        
        plot_feature_importance(classifier.model, feature_names, top_n=30)
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
    
    # Save model comparison results
    save_model_comparison_report(model_results, data_loader.class_names, 
                                 final_results, classifier.model_name)
    
    # Save final model
    model_path = "emg_classifier_best_model.pkl"
    classifier.save_model(model_path)
    print(f"\nBest model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - emg_classifier_best_model.pkl (trained model)")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - class_distribution.png")
    print("  - model_comparison.png")
    print("  - emg_signals.png")
    print("  - model_comparison_report.txt")
    
    return classifier, model_results, final_results


def run_cross_validation_with_models(data_dir: str, n_trials: int = 49, 
                                     n_reps: int = 5,
                                     window_size_ms: int = 100, 
                                     overlap: float = 0.5,
                                     sampling_rate: int = 500,
                                     test_top_n: int = 3):
    """
    Run leave-one-trial-out CV with top N models from initial comparison
    
    Args:
        data_dir: Directory containing EMG data
        n_trials: Number of trials
        n_reps: Number of repetitions per trial
        window_size_ms: Window size in milliseconds
        overlap: Window overlap ratio
        sampling_rate: Sampling rate in Hz
        test_top_n: Number of top models to test in CV
    """
    print("="*80)
    print("LEAVE-ONE-TRIAL-OUT CROSS-VALIDATION WITH MODEL COMPARISON")
    print("="*80)
    
    # Initialize
    data_loader = EMGDataLoader(data_dir)
    classifier = EMGClassifier(
        window_size_ms=window_size_ms,
        overlap=overlap,
        sampling_rate=sampling_rate
    )
    
    # Load all data
    print("\nLoading data...")
    data_list, labels, metadata = data_loader.load_dataset(n_trials, n_reps)
    
    # First, do a quick model comparison on one fold
    print("\n" + "="*80)
    print("INITIAL MODEL SCREENING (Using Trial 10 as test)")
    print("="*80)
    
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    for data, label, meta in zip(data_list, labels, metadata):
        if meta['trial'] == 10:
            test_data.append(data)
            test_labels.append(label)
        else:
            train_data.append(data)
            train_labels.append(label)
    
    X_train, y_train = classifier.prepare_data(train_data, train_labels)
    X_test, y_test = classifier.prepare_data(test_data, test_labels)
    
    # Train all models for screening
    screening_results = classifier.train_and_compare_models(
        X_train, y_train, X_test, y_test,
        include_ensembles=True, cv_folds=3
    )
    
    # Select top N models
    sorted_results = sorted(screening_results, 
                           key=lambda x: x['test_accuracy'], 
                           reverse=True)
    top_models = [r['model_name'] for r in sorted_results[:test_top_n]]
    
    print(f"\nTop {test_top_n} models selected for CV:")
    for i, name in enumerate(top_models, 1):
        print(f"  {i}. {name}")
    
    # Now run CV with top models
    print("\n" + "="*80)
    print(f"RUNNING CV WITH TOP {test_top_n} MODELS")
    print("="*80)
    
    cv_results = {model_name: [] for model_name in top_models}
    
    for test_trial in range(1, n_trials + 1):
        print(f"\n{'='*80}")
        print(f"FOLD {test_trial}: Testing on Trial {test_trial}")
        print(f"{'='*80}")
        
        # Split data
        train_data, train_labels = [], []
        test_data, test_labels = [], []
        
        for data, label, meta in zip(data_list, labels, metadata):
            if meta['trial'] == test_trial:
                test_data.append(data)
                test_labels.append(label)
            else:
                train_data.append(data)
                train_labels.append(label)
        
        X_train, y_train = classifier.prepare_data(train_data, train_labels)
        X_test, y_test = classifier.prepare_data(test_data, test_labels)
        
        # Train and test each top model
        for model_name in top_models:
            print(f"\nTraining {model_name}...")
            
            # Create new classifier instance
            fold_classifier = EMGClassifier(
                window_size_ms=window_size_ms,
                overlap=overlap,
                sampling_rate=sampling_rate
            )
            
            # Get the specific model from stored models
            from emg_classification_pipeline import ModelFactory
            all_models = ModelFactory.get_base_models()
            if model_name in all_models:
                model = all_models[model_name]
            else:
                # It's an ensemble
                ensembles = ModelFactory.get_ensemble_models(all_models)
                model = ensembles[model_name]
            
            # Train
            X_train_scaled = fold_classifier.scaler.fit_transform(X_train)
            X_test_scaled = fold_classifier.scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            fold_classifier.model = model
            fold_classifier.model_name = model_name
            fold_classifier.is_trained = True
            
            # Evaluate
            fold_results = evaluate_model(
                fold_classifier, X_test, y_test, data_loader.class_names
            )
            fold_results['test_trial'] = test_trial
            cv_results[model_name].append(fold_results)
    
    # Compare CV results across models
    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<30} {'Mean Acc':<15} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*80)
    
    best_model = None
    best_score = 0
    
    for model_name in top_models:
        accuracies = [r['accuracy'] for r in cv_results[model_name]]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        print(f"{model_name:<30} {mean_acc:<15.4f} {std_acc:<10.4f} "
              f"{min_acc:<10.4f} {max_acc:<10.4f}")
        
        if mean_acc > best_score:
            best_score = mean_acc
            best_model = model_name
    
    print("\n" + "="*80)
    print(f"BEST MODEL FROM CV: {best_model}")
    print(f"Mean Accuracy: {best_score:.4f}")
    print("="*80)
    
    # Visualize detailed CV results
    plot_cv_detailed(cv_results, data_loader.class_names)
    
    return cv_results, best_model


def quick_train_and_test(data_dir: str, test_trial: int = 10,
                         include_ensembles: bool = True,
                         optimize: bool = False):
    """
    Quick training on 9 trials and testing on 1 trial with model comparison
    
    Args:
        data_dir: Directory containing EMG data
        test_trial: Trial number to use for testing
        include_ensembles: Whether to include ensemble models
        optimize: Whether to optimize the best model
    """
    print("\n" + "="*80)
    print(f"QUICK TRAIN & TEST WITH MODEL COMPARISON")
    print(f"Testing on Trial {test_trial}")
    print("="*80)
    
    # Initialize
    data_loader = EMGDataLoader(data_dir)
    classifier = EMGClassifier()
    
    # Load data
    print("\nLoading data...")
    data_list, labels, metadata = data_loader.load_dataset(10, 5)
    
    # Split by trial
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    for data, label, meta in zip(data_list, labels, metadata):
        if meta['trial'] == test_trial:
            test_data.append(data)
            test_labels.append(label)
        else:
            train_data.append(data)
            train_labels.append(label)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Prepare features
    print("\nExtracting features...")
    X_train, y_train = classifier.prepare_data(train_data, train_labels)
    X_test, y_test = classifier.prepare_data(test_data, test_labels)
    
    # Train and compare models
    print("\nTraining and comparing models...")
    model_results = classifier.train_and_compare_models(
        X_train, y_train,
        X_test, y_test,
        include_ensembles=include_ensembles,
        cv_folds=5
    )
    
    # Visualize
    plot_model_comparison(model_results)
    
    # Optimize if requested
    if optimize:
        print("\nOptimizing best model...")
        try:
            classifier.optimize_best_model(X_train, y_train)
        except Exception as e:
            print(f"Could not optimize: {e}")
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    results = evaluate_model(classifier, X_test, y_test, data_loader.class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        data_loader.class_names,
        title=f'Confusion Matrix - {classifier.model_name} (Test Trial {test_trial})'
    )
    
    return classifier, model_results, results


def test_realtime_prediction(model_path: str, test_data_path: str):
    """
    Test real-time prediction on a single file
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data CSV file
    """
    print("\n" + "="*80)
    print("REAL-TIME PREDICTION TEST")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    classifier = EMGClassifier()
    classifier.load_model(model_path)
    
    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path).values
    print(f"Test data shape: {test_data.shape}")
    
    # Simulate real-time prediction
    window_size = 50  # 100ms at 500Hz
    step_size = 25    # 50% overlap
    
    predictions = []
    probabilities = []
    
    print("\nRunning real-time predictions...")
    for start in range(0, len(test_data) - window_size + 1, step_size):
        end = start + window_size
        window = test_data[start:end, :]
        
        pred, proba = classifier.predict_realtime(window)
        predictions.append(pred)
        probabilities.append(proba)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    print(f"\nGenerated {len(predictions)} predictions")
    print("\nPrediction distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    
    class_names = {0: 'extended_rest', 1: 'flexing', 
                   2: 'flexed_rest', 3: 'extending'}
    
    for class_id, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        print(f"  Class {class_id} ({class_names[class_id]}): "
              f"{count} ({percentage:.1f}%)")
    
    # Calculate average confidence
    max_probas = np.max(probabilities, axis=1)
    print(f"\nAverage confidence: {np.mean(max_probas):.3f}")
    print(f"Min confidence: {np.min(max_probas):.3f}")
    print(f"Max confidence: {np.max(max_probas):.3f}")
    
    return predictions, probabilities


def save_model_comparison_report(model_results: list, class_names: dict,
                                 final_results: dict, best_model_name: str,
                                 output_file: str = 'model_comparison_report.txt'):
    """
    Save detailed model comparison report
    
    Args:
        model_results: List of model comparison results
        class_names: Dictionary mapping class IDs to names
        final_results: Final evaluation results
        best_model_name: Name of the best model
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EMG MOVEMENT CLASSIFICATION - MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Model comparison
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<30} {'Train Acc':<12} {'Test Acc':<12} {'CV Score':<15} {'Time (s)':<10}\n")
        f.write("-"*80 + "\n")
        
        for result in sorted(model_results, key=lambda x: x.get('test_accuracy', 0), reverse=True):
            test_acc_str = f"{result['test_accuracy']:.4f}" if result['test_accuracy'] is not None else "N/A"
            cv_str = f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}"
            f.write(f"{result['model_name']:<30} {result['train_accuracy']:<12.4f} "
                   f"{test_acc_str:<12} {cv_str:<15} {result['train_time']:<10.2f}\n")
        
        # Best model
        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write("="*80 + "\n\n")
        
        # Final evaluation
        f.write("FINAL EVALUATION ON TEST SET\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:  {final_results['accuracy']:.4f}\n")
        f.write(f"Precision: {final_results['precision']:.4f}\n")
        f.write(f"Recall:    {final_results['recall']:.4f}\n")
        f.write(f"F1-Score:  {final_results['f1']:.4f}\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-"*80 + "\n")
        for i in range(len(class_names)):
            f.write(f"\nClass {i} ({class_names[i]}):\n")
            f.write(f"  Precision: {final_results['per_class_metrics']['precision'][i]:.4f}\n")
            f.write(f"  Recall: {final_results['per_class_metrics']['recall'][i]:.4f}\n")
            f.write(f"  F1-Score: {final_results['per_class_metrics']['f1'][i]:.4f}\n")
            f.write(f"  Support: {final_results['per_class_metrics']['support'][i]}\n")
        
        # Confusion matrix
        f.write("\nCONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        f.write(str(final_results['confusion_matrix']) + "\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Model comparison report saved to {output_file}")


def test_window_sizes(data_dir, window_sizes=[50, 75, 100, 125], test_trial=10):
    from emg_classification_pipeline import EMGDataLoader, EMGClassifier, evaluate_model
    data_loader = EMGDataLoader(data_dir)
    
    data_list, labels, metadata = data_loader.load_dataset(10, 5)
    train_data, train_labels, test_data, test_labels = [], [], [], []
    for data, label, meta in zip(data_list, labels, metadata):
        if meta['trial'] == test_trial:
            test_data.append(data)
            test_labels.append(label)
        else:
            train_data.append(data)
            train_labels.append(label)

    results = []
    for w in window_sizes:
        print(f"\n=== Testing Window Size {w} ms ===")
        clf = EMGClassifier(window_size_ms=w, overlap=0.5, sampling_rate=500)
        X_train, y_train = clf.prepare_data(train_data, train_labels)
        X_test, y_test = clf.prepare_data(test_data, test_labels)
        print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
        clf.train_and_compare_models(X_train, y_train, X_test, y_test, include_ensembles=False, cv_folds=3)
        results.append({
            "window_size": w,
            "best_model": clf.model_name,
            "test_acc": clf.model_scores.get("test_accuracy", 0),
            "cv_mean": clf.model_scores.get("cv_mean", 0)
        })

    print("\n=== Summary ===")
    print(f"{'Window(ms)':<10} {'Best Model':<25} {'Test Acc':<10} {'CV Mean':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['window_size']:<10} {r['best_model']:<25} {r['test_acc']:<10.4f} {r['cv_mean']:<10.4f}")
    return results



if __name__ == "__main__":
    """
    Main execution
    
    Usage:
        python main_analysis.py [data_directory]
    """
    
    # Get data directory from command line or use default
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        DATA_DIR = "./emg_data"
    
    # Check if directory exists
    if not Path(DATA_DIR).exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("\nPlease provide the path to your EMG data directory:")
        print("  python main_analysis.py /path/to/emg/data")
        sys.exit(1)
    
    print(f"Using data directory: {DATA_DIR}\n")
    
    # Choose analysis mode
    print("Select analysis mode:")
    print("  1. Complete analysis with model comparison (recommended)")
    print("  2. Quick train & test with model comparison")
    print("  3. Leave-one-trial-out CV with top models")
    print("  4. Test real-time prediction")
    print("  5. Window size optimization")

    choice = input("\nEnter choice (1/2/3/4/5) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        # Complete analysis with model comparison
        include_ensembles = input("Include ensemble models? (y/n) [default: y]: ").strip().lower() != 'n'
        optimize_best = input("Optimize best model? (y/n) [default: y]: ").strip().lower() != 'n'
        
        classifier, model_results, final_results = run_complete_analysis(
            DATA_DIR,
            include_ensembles=include_ensembles,
            optimize_best=optimize_best
        )
        
    elif choice == "2":
        # Quick train and test
        test_trial = input("Enter test trial number (1-10) [default: 10]: ").strip()
        test_trial = int(test_trial) if test_trial else 10
        
        include_ensembles = input("Include ensemble models? (y/n) [default: y]: ").strip().lower() != 'n'
        optimize = input("Optimize best model? (y/n) [default: n]: ").strip().lower() == 'y'
        
        classifier, model_results, results = quick_train_and_test(
            DATA_DIR, 
            test_trial, 
            include_ensembles,
            optimize
        )
        
    elif choice == "3":
        # CV with top models
        top_n = input("Number of top models to test (1-5) [default: 3]: ").strip()
        top_n = int(top_n) if top_n else 3
        
        cv_results, best_model = run_cross_validation_with_models(
            DATA_DIR,
            test_top_n=top_n
        )
        
    elif choice == "4":
        # Real-time prediction test
        model_path = input("Enter model path [emg_classifier_best_model.pkl]: ").strip()
        model_path = model_path or "emg_classifier_best_model.pkl"
        
        test_file = input("Enter test data CSV path: ").strip()
        if test_file and Path(test_file).exists():
            predictions, probabilities = test_realtime_prediction(model_path, test_file)
        else:
            print(f"Error: Test file not found: {test_file}")
    
    elif choice == "5":
        # Window size optimization mode
        window_sizes = input("Enter window sizes in ms (comma-separated, e.g., 50,75,100,125) [default: 50,75,100,125]: ").strip()
        if window_sizes:
            window_sizes = [int(x.strip()) for x in window_sizes.split(",")]
        else:
            window_sizes = [50, 75, 100, 125]

        test_trial = input("Enter test trial number (1-10) [default: 10]: ").strip()
        test_trial = int(test_trial) if test_trial else 10

        test_window_sizes(DATA_DIR, window_sizes, test_trial)

    else:
        print("Invalid choice")
        sys.exit(1)
    
    print("\nDone!")