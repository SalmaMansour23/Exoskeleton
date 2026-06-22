"""
EMG Movement Classification Pipeline - Multi-Model Version
Complete pipeline with support for multiple model types and automatic selection
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
import time

warnings.filterwarnings('ignore')


class EMGPreprocessor:
    """Handles EMG signal preprocessing and filtering"""
    
    def __init__(self, sampling_rate: int = 500):
        """
        Args:
            sampling_rate: Sampling frequency in Hz (default 500 Hz)
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Design bandpass filter (20-200 Hz)
        self.bp_low = 20 / self.nyquist
        self.bp_high = min(200, self.nyquist * 0.9) / self.nyquist
        
        if self.bp_high >= 1.0:
            self.bp_high = 0.9
            
        self.b_bp, self.a_bp = signal.butter(
            4, [self.bp_low, self.bp_high], btype='band'
        )
        
        # Design notch filter (50 Hz)
        self.notch_freqs = [50]
        self.notch_filters = []
        for freq in self.notch_freqs:
            if freq < self.nyquist:
                w0 = freq / self.nyquist
                Q = 30.0
                b_notch, a_notch = signal.iirnotch(w0, Q)
                self.notch_filters.append((b_notch, a_notch))
    
    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter"""
        return signal.filtfilt(self.b_bp, self.a_bp, data, axis=0)
    
    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filters for power line interference"""
        filtered = data.copy()
        for b_notch, a_notch in self.notch_filters:
            filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=0)
        return filtered
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw EMG data (samples x channels)
            
        Returns:
            Preprocessed EMG data
        """
        if data.shape[0] <= 27:
            raise ValueError(f"Not enough samples ({data.shape[0]}) for filtering (need >27).")
        filtered = self.apply_bandpass(data)
        filtered = self.apply_notch(filtered)
        return filtered


class EMGFeatureExtractor:
    """Extracts time-domain features from EMG windows"""
    
    def __init__(self, window_size_ms: int = 100, overlap: float = 0.5, 
                 sampling_rate: int = 500):
        """
        Args:
            window_size_ms: Window size in milliseconds
            overlap: Overlap ratio between windows (0-1)
            sampling_rate: Sampling frequency in Hz
        """
        self.window_size = int(window_size_ms * sampling_rate / 1000)
        self.step_size = int(self.window_size * (1 - overlap))
        self.sampling_rate = sampling_rate
    
    def create_windows(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows from continuous data"""
        n_samples, n_channels = data.shape
        windows = []
        
        for start in range(0, n_samples - self.window_size + 1, self.step_size):
            end = start + self.window_size
            windows.append(data[start:end, :])
        
        return np.array(windows)
    
    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract time-domain features from a single window"""
        features = []
        
        for ch in range(window.shape[1]):
            signal_ch = window[:, ch]
            
            # Mean Absolute Value
            mav = np.mean(np.abs(signal_ch))
            
            # Root Mean Square
            rms = np.sqrt(np.mean(signal_ch ** 2))
            
            # Waveform Length
            wl = np.sum(np.abs(np.diff(signal_ch)))
            
            # Zero Crossing Rate
            zcr = np.sum(np.diff(np.sign(signal_ch)) != 0) / len(signal_ch)
            
            # Slope Sign Changes
            diff_signal = np.diff(signal_ch)
            ssc = np.sum(np.diff(np.sign(diff_signal)) != 0) / len(signal_ch)
            
            # Standard Deviation
            std = np.std(signal_ch)
            
            # Variance
            var = np.var(signal_ch)
            
            # Kurtosis
            kurt = kurtosis(signal_ch)
            
            # Skewness
            skewness = skew(signal_ch)
            
            features.extend([mav, rms, wl, zcr, ssc, std, var, kurt, skewness])
        
        return np.array(features)
    
    def extract_features_from_windows(self, windows: np.ndarray) -> np.ndarray:
        """Extract features from all windows"""
        features = []
        for window in windows:
            features.append(self.extract_features(window))
        return np.array(features)


class EMGDataLoader:
    """Loads and organizes EMG data from file structure"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Root directory containing EMG data files
        """
        self.data_dir = Path(data_dir)
        self.class_names = {
            0: 'extended_rest',
            1: 'flexing',
            2: 'flexed_rest',
            3: 'extending'
        }
        self.class_name_to_id = {v: k for k, v in self.class_names.items()}
    
    def load_file(self, filepath: Path) -> np.ndarray:
        """Load a single CSV file"""
        try:
            data = pd.read_csv(filepath)
            return data.values
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_dataset(self, n_trials: int = 49, n_reps: int = 5) -> Tuple[List, List, List]:
        """Load complete dataset"""
        data_list = []
        labels = []
        metadata = []
        
        for trial in range(1, n_trials + 1):
            for rep in range(1, n_reps + 1):
                for class_id, class_name in self.class_names.items():
                    filename = f"trial_{trial}_rep{rep}_{class_id}_{class_name}.csv"
                    filepath = self.data_dir / filename
                    
                    if filepath.exists():
                        data = self.load_file(filepath)
                        if data is not None:
                            data_list.append(data)
                            labels.append(class_id)
                            metadata.append({
                                'trial': trial,
                                'rep': rep,
                                'class': class_id,
                                'class_name': class_name
                            })
                    else:
                        print(f"Warning: File not found: {filename}")
        
        return data_list, labels, metadata


class ModelFactory:
    """Factory class for creating different model types"""
    
    @staticmethod
    def get_base_models() -> Dict:
        """Get dictionary of base models with default parameters"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42, 
                n_jobs=-1
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', 
                C=10, 
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'SVM (Linear)': SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis(),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=5,
                random_state=42
            )
        }
        return models
    
    @staticmethod
    def get_ensemble_models(base_models: Dict) -> Dict:
        """Create ensemble models from base models"""
        ensembles = {}
        
        # Voting Classifier - Top performers
        if 'Random Forest' in base_models and 'XGBoost' in base_models and 'SVM (RBF)' in base_models:
            ensembles['Voting (RF+XGB+SVM)'] = VotingClassifier(
                estimators=[
                    ('rf', base_models['Random Forest']),
                    ('xgb', base_models['XGBoost']),
                    ('svm', base_models['SVM (RBF)'])
                ],
                voting='soft',
                n_jobs=-1
            )
        
        # Voting Classifier - All tree-based
        tree_models = []
        for name, model in base_models.items():
            if any(x in name for x in ['Forest', 'XGBoost', 'Gradient', 'Extra', 'AdaBoost']):
                tree_models.append((name.lower().replace(' ', '_'), model))
        
        if len(tree_models) >= 3:
            ensembles['Voting (All Trees)'] = VotingClassifier(
                estimators=tree_models[:5],  # Limit to 5 to avoid overfitting
                voting='soft',
                n_jobs=-1
            )
        
        return ensembles
    
    @staticmethod
    def get_optimized_model_params() -> Dict:
        """Get parameter grids for hyperparameter optimization"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM (RBF)': {
                'C': [1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        return param_grids


class EMGClassifier:
    """Main classifier for EMG movement prediction with multi-model support"""
    
    def __init__(self, window_size_ms: int = 100, overlap: float = 0.5,
                 sampling_rate: int = 500):
        """
        Args:
            window_size_ms: Window size in milliseconds
            overlap: Overlap ratio between windows
            sampling_rate: Sampling frequency in Hz
        """
        self.preprocessor = EMGPreprocessor(sampling_rate)
        self.feature_extractor = EMGFeatureExtractor(
            window_size_ms, overlap, sampling_rate
        )
        self.scaler = StandardScaler()
        self.model = None
        self.model_name = None
        self.all_models = {}
        self.model_scores = {}
        self.is_trained = False
        
    def prepare_data(self, data_list: List[np.ndarray], 
                    labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess and extract features from raw data"""
        X_all = []
        y_all = []
        
        for data, label in zip(data_list, labels):
            if data.shape[0] <= 27:
                print(f"⚠️ Skipping file with only {data.shape[0]} samples (too short for filtering).")
                continue
            try:
                preprocessed = self.preprocessor.preprocess(data)
            except Exception as e:
                print(f"⚠️ Error processing file (skipped): {e}")
                continue

            windows = self.feature_extractor.create_windows(preprocessed)
            features = self.feature_extractor.extract_features_from_windows(windows)
            
            X_all.append(features)
            y_all.extend([label] * len(features))
        
        X = np.vstack(X_all)
        y = np.array(y_all)
        
        return X, y
    
    def train_and_compare_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray = None, y_test: np.ndarray = None,
                                 include_ensembles: bool = True,
                                 cv_folds: int = 5) -> Dict:
        """
        Train multiple models and compare their performance
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features for validation
            y_test: Optional test labels for validation
            include_ensembles: Whether to include ensemble models
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with model comparison results
        """
        print("\n" + "="*80)
        print("TRAINING AND COMPARING MULTIPLE MODELS")
        print("="*80)
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        # Get all models
        base_models = ModelFactory.get_base_models()
        
        results = []
        
        # Train and evaluate base models
        print("\n[1/2] Training base models...")
        print("-"*80)
        
        for i, (name, model) in enumerate(base_models.items(), 1):
            print(f"\n[{i}/{len(base_models)}] Training {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                train_time = time.time() - start_time
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                           cv=cv_folds, n_jobs=-1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Training accuracy
                train_pred = model.predict(X_train_scaled)
                train_acc = accuracy_score(y_train, train_pred)
                
                # Test accuracy (if test set provided)
                test_acc = None
                if X_test is not None and y_test is not None:
                    test_pred = model.predict(X_test_scaled)
                    test_acc = accuracy_score(y_test, test_pred)
                
                # Store model
                self.all_models[name] = model
                
                result = {
                    'model_name': name,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'train_time': train_time
                }
                results.append(result)
                
                print(f"  Training Accuracy: {train_acc:.4f}")
                print(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
                if test_acc is not None:
                    print(f"  Test Accuracy: {test_acc:.4f}")
                print(f"  Training Time: {train_time:.2f}s")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue
        
        # Train and evaluate ensemble models
        if include_ensembles:
            print("\n[2/2] Training ensemble models...")
            print("-"*80)
            
            ensemble_models = ModelFactory.get_ensemble_models(base_models)
            
            for i, (name, model) in enumerate(ensemble_models.items(), 1):
                print(f"\n[{i}/{len(ensemble_models)}] Training {name}...")
                start_time = time.time()
                
                try:
                    model.fit(X_train_scaled, y_train)
                    train_time = time.time() - start_time
                    
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                               cv=cv_folds, n_jobs=-1)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    train_pred = model.predict(X_train_scaled)
                    train_acc = accuracy_score(y_train, train_pred)
                    
                    test_acc = None
                    if X_test is not None and y_test is not None:
                        test_pred = model.predict(X_test_scaled)
                        test_acc = accuracy_score(y_test, test_pred)
                    
                    self.all_models[name] = model
                    
                    result = {
                        'model_name': name,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'train_time': train_time
                    }
                    results.append(result)
                    
                    print(f"  Training Accuracy: {train_acc:.4f}")
                    print(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
                    if test_acc is not None:
                        print(f"  Test Accuracy: {test_acc:.4f}")
                    print(f"  Training Time: {train_time:.2f}s")
                    
                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    continue
        
        # Select best model
        self._select_best_model(results, X_test is not None)
        
        return results
    
    def _select_best_model(self, results: List[Dict], has_test_set: bool):
        """Select the best model based on results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Sort by appropriate metric
        if has_test_set:
            results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
            print("\nRanked by Test Accuracy:")
        else:
            results_sorted = sorted(results, key=lambda x: x['cv_mean'], reverse=True)
            print("\nRanked by Cross-Validation Score:")
        
        print("-"*80)
        print(f"{'Rank':<6} {'Model':<30} {'Train Acc':<12} {'Test Acc':<12} {'CV Score':<15} {'Time (s)':<10}")
        print("-"*80)
        
        for i, result in enumerate(results_sorted, 1):
            test_acc_str = f"{result['test_accuracy']:.4f}" if result['test_accuracy'] is not None else "N/A"
            cv_str = f"{result['cv_mean']:.4f}±{result['cv_std']:.4f}"
            print(f"{i:<6} {result['model_name']:<30} {result['train_accuracy']:<12.4f} "
                  f"{test_acc_str:<12} {cv_str:<15} {result['train_time']:<10.2f}")
        
        # Select best model
        best_result = results_sorted[0]
        self.model = self.all_models[best_result['model_name']]
        self.model_name = best_result['model_name']
        self.model_scores = best_result
        self.is_trained = True
        
        print("\n" + "="*80)
        print(f"BEST MODEL SELECTED: {self.model_name}")
        print("="*80)
        print(f"Train Accuracy: {best_result['train_accuracy']:.4f}")
        if best_result['test_accuracy'] is not None:
            print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
        print(f"CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        print("="*80)
    
    def optimize_best_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           cv_folds: int = 3) -> None:
        """Optimize hyperparameters of the best model"""
        if not self.is_trained or self.model_name is None:
            raise ValueError("Must train and compare models first!")
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZING {self.model_name}")
        print(f"{'='*80}")
        
        param_grids = ModelFactory.get_optimized_model_params()
        
        if self.model_name not in param_grids:
            print(f"No parameter grid defined for {self.model_name}")
            return
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Get base model class
        base_models = ModelFactory.get_base_models()
        if self.model_name not in base_models:
            print(f"Cannot optimize ensemble model {self.model_name}")
            return
        
        base_model = base_models[self.model_name]
        param_grid = param_grids[self.model_name]
        
        print(f"Searching parameter space...")
        print(f"Parameters to optimize: {list(param_grid.keys())}")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update model
        self.model = grid_search.best_estimator_
        self.all_models[self.model_name] = self.model
        
        print(f"\nModel {self.model_name} optimized successfully!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_realtime(self, data: np.ndarray) -> Tuple[int, np.ndarray]:
        """Real-time prediction from raw EMG data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        preprocessed = self.preprocessor.preprocess(data)
        features = self.feature_extractor.extract_features(preprocessed)
        features = features.reshape(1, -1)
        
        pred = self.predict(features)[0]
        proba = self.predict_proba(features)[0]
        
        return pred, proba
    
    def save_model(self, filepath: str) -> None:
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_scores': self.model_scores,
            'all_models': self.all_models,
            'scaler': self.scaler,
            'preprocessor': self.preprocessor,
            'feature_extractor': self.feature_extractor
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data.get('model_name', 'Unknown')
        self.model_scores = model_data.get('model_scores', {})
        self.all_models = model_data.get('all_models', {})
        self.scaler = model_data['scaler']
        self.preprocessor = model_data['preprocessor']
        self.feature_extractor = model_data['feature_extractor']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        print(f"Active model: {self.model_name}")


def evaluate_model(classifier: EMGClassifier, X_test: np.ndarray, 
                   y_test: np.ndarray, class_names: Dict) -> Dict:
    """Comprehensive model evaluation"""
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_test, y_pred, average=None)
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\nModel: {classifier.model_name}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    for i in range(len(class_names)):
        print(f"\nClass {i} ({class_names[i]}):")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1-Score: {f1_per_class[i]:.4f}")
        print(f"  Support: {support_per_class[i]}")
    
    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    print(cm)
    
    print("\n" + "-"*60)
    print("Detailed Classification Report:")
    print("-"*60)
    target_names = [class_names[i] for i in sorted(class_names.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_metrics': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support_per_class
        }
    }


if __name__ == "__main__":
    print("EMG Classification Pipeline - Multi-Model Version")
    print("Import this module in main_analysis.py to use")