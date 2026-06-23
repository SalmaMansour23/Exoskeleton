# EEG and EMG Intent Detection

This repository contains two related pipelines for intent detection in an exoskeleton research project:

- **EEG**: a 3-class movement intent classification pipeline for `NoMovement`, `Flexing`, and `Extending` trained on EEG recordings.
- **EMG**: a 4-class movement intent classification pipeline trained on EMG recordings for offline analysis and real-time prediction.

## Repository Layout

- `EEG/code/train_3class_model.ipynb` - training notebook for the EEG 3-class model.
- `EEG/code/live_3class_detector.py` - live EEG inference script.
- `EEG/code/results & model/` - saved EEG model, scaler, label encoder, metadata, and evaluation plots.
- `EMG/code/emg_classification_pipeline.py` - main EMG training pipeline.
- `EMG/code/main_analysis.py` - EMG analysis and evaluation script.
- `EMG/code/real_time_predictor.py` - real-time EMG prediction script.
- `EMG/code/results & model/` - trained EMG model and performance plots/reports.
- `EEG/EEG_Data/` and `EMG/EMG_Data/` - training and testing data.

## Models and Training Data

The EEG pipeline was trained with XGBoost on preprocessed EEG features extracted from multiple channels and frequency bands. The EMG pipeline compares several classical machine-learning models and saves the best-performing classifier for downstream use.

## Notes

- The project includes both offline training and real-time prediction code.
- Large data files and generated model artifacts are kept inside the EEG and EMG folders.
- The saved models can be reused without retraining if the preprocessing steps are kept the same.