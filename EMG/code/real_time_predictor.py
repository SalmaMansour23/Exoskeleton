"""
Real-time EMG Movement Prediction using MindRove and EMGClassifier pipeline
Compatible with emg_classifier_best_model.pkl saved from main_analysis.py
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='mindrove')

import time
import joblib
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

# === Load your trained pipeline model bundle ===
BUNDLE_PATH = "emg_classifier_best_model.pkl"
bundle = joblib.load(BUNDLE_PATH)
classifier = bundle["preprocessor"]  # This line is just for quick reference check
print("[INFO] Model bundle keys:", bundle.keys())

# Load full EMGClassifier object components
model = bundle["model"]
scaler = bundle["scaler"]
preprocessor = bundle["preprocessor"]
feature_extractor = bundle["feature_extractor"]

# === Config ===
SAMPLING_RATE = 500
WINDOW_MS = 100
WINDOW_SIZE = int(SAMPLING_RATE * (WINDOW_MS / 1000))  # 50 samples
REFRESH_TIME = 0.1
PRED_HISTORY = deque(maxlen=5)

CLASS_NAMES = {0: "extended_rest", 1: "flexing", 2: "flexed_rest", 3: "extending"}
current_state = "extended_rest"

# === GUI ===
class StateWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time EMG Prediction (Pipeline Model)")
        self.setGeometry(100, 100, 420, 150)
        self.label = QtWidgets.QLabel("Initializing...", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 50px;")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.show()

    def update_state(self, text):
        self.label.setText(f"State: {text}")

app = QtWidgets.QApplication([])
window = StateWindow()


def predict_from_raw_window(raw_window: np.ndarray):
    """Applies the same preprocessing + feature extraction as training."""
    # raw_window shape: (channels, samples)
    raw_window = raw_window.T  # → (samples, channels)
    try:
        preprocessed = preprocessor.preprocess(raw_window)
        features = feature_extractor.extract_features(preprocessed).reshape(1, -1)
        X_scaled = scaler.transform(features)
        pred = model.predict(X_scaled)[0]
        return pred
    except Exception as e:
        print(f"[WARN] Skipped window: {e}")
        return None


# === MindRove Connection ===
BoardShim.enable_dev_board_logger()
params = MindRoveInputParams()
board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)

try:
    print("[INFO] Connecting to MindRove...")
    board.prepare_session()
    board.start_stream()
    print("[INFO] Streaming started...")

    while True:
        QtWidgets.QApplication.processEvents()

        data = board.get_current_board_data(WINDOW_SIZE)
        if data.shape[1] < WINDOW_SIZE:
            continue

        emg_channels = BoardShim.get_exg_channels(BoardIds.MINDROVE_WIFI_BOARD)
        emg_data = data[emg_channels, :]  # shape: (8, 50)

        prediction = predict_from_raw_window(emg_data)
        if prediction is None:
            continue

        pred_label = CLASS_NAMES.get(prediction, str(prediction))
        PRED_HISTORY.append(pred_label)

        # majority voting smoothing
        most_common = max(set(PRED_HISTORY), key=PRED_HISTORY.count)

        if most_common != current_state:
            print(f"[ACTION] {most_common} (from {current_state})")
            current_state = most_common
        else:
            print(f"[STATE] {current_state}")

        window.update_state(current_state)
        time.sleep(REFRESH_TIME)

except KeyboardInterrupt:
    print("\n[STOP] Interrupted by user.")
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    if board.is_prepared():
        board.release_session()
        print("[INFO] Session released")
