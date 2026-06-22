"""
live_3class_detector.py — Marker-triggered 3-class BCI.

Classifies every trial into one of three classes:
  NoMovement  — arm is resting (neither flexing nor extending)
  Flexing     — arm is flexing
  Extending   — arm is extending

TRIGGER
-------
  Waits for "PostureStart_Trial_N" markers from experiment_calibration.py
  (sent via XonMarkerStream LSL outlet).  On each marker, collects a 3-second
  Full Execution epoch and runs the trained xgboost 3-class model.

SERIAL OUTPUT
-------------
  A trigger is sent to the exoskeleton controller ONLY when the prediction
  is Flexing or Extending.  NoMovement predictions are suppressed.

  Message format:  "ML_FLEXING\\n"  /  "ML_EXTENDING\\n"

  If --serial-port is omitted the trigger is printed to stdout instead.

FEATURE VECTOR (30 per trial)
------------------------------
  [0:28]  28 frequency features
            7 channels × 3 bands = 21 bandpower values (mu, beta, low_gamma)
            2 C3–C4 lateralisation values (mu diff, beta diff)
            2 motor/frontal ratios
            3 time-domain std values (C3, Cz, C4)
  [28]    ERD_dB — mu+beta power ratio: first 1 s vs last 2 s of epoch at C3+C4
            Negative dB = desynchronisation = movement; ~0 = rest
  [29]    cycle_pos / 3.0 — (trial_N − 1) % 4, normalised
            Encodes position in the repeating 4-trial block (legitimate here
            because all 3 classes span all 4 positions in training)

CYCLE POSITION
--------------
  Pattern A:  0 = Flex, 1 = RestFlex, 2 = Extend, 3 = RestExtend
  Pattern B:  0 = Extend, 1 = RestExtend, 2 = Flex, 3 = RestFlex

  The model was trained on both patterns, so cycle_pos is informative but
  not deterministic

Requirements
------------
  pip install xgboost scikit-learn scipy numpy pylsl pyserial

Usage
  python live_3class_detector.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import deque
from typing import Optional

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import numpy as np
from scipy import signal as sp_signal

# NumPy 2.0 removed trapz → support both
try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz

try:
    from pylsl import StreamInlet, resolve_streams
except ImportError:
    sys.exit("pylsl not found.  Run:  pip install pylsl")


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
CHANNEL_NAMES       = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]
N_CHANNELS          = 7
BUFFER_SECS         = 15.0    # rolling EEG buffer (must cover baseline + epoch)
MARKER_STREAM_NAME  = "XonMarkerStream"

# Epoch window: 3 s after PostureStart (matches train_3class_model.ipynb)
EPOCH_SECS          = 3.0

# Baseline: 1 s window from −5 s to −4 s relative to PostureStart
BASELINE_PRE_SECS   = 5.0     # seconds BEFORE PostureStart to start baseline
BASELINE_POST_SECS  = 4.0     # seconds BEFORE PostureStart to end baseline

# ERD feature: first N seconds of epoch = pseudo-baseline
ERD_BASELINE_SECS   = 1.0

# Artifact rejection: max peak-to-peak amplitude at motor channels (µV)
ARTIFACT_THRESHOLD  = 1500.0


# ── Signal processing helpers ─────────────────────────────────────────────────

def _bandpass(data: np.ndarray, low: float, high: float,
              fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, data, axis=0)


def _notch(data: np.ndarray, freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    b, a = sp_signal.iirnotch(freq, Q, fs)
    return sp_signal.filtfilt(b, a, data, axis=0)


def _bandpower(psd: np.ndarray, freqs: np.ndarray, band: tuple) -> float:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(_trapz(psd[mask], freqs[mask]))


# ── Feature extraction (must match train_3class_model.ipynb exactly) ──────────

def extract_freq_features(
    epoch: np.ndarray,
    fs: float,
    freq_bands: dict,
) -> np.ndarray:
    """
    28 frequency features from a preprocessed (N_samples × 7) epoch.
    Channel order: F3, F4, C3, Cz, C4, P3, P4.
    """
    nperseg = min(256, epoch.shape[0])
    feats:  list[float]      = []
    psds:   list[tuple]      = []

    # 7 channels × 3 bands = 21 bandpower features
    for ch in range(7):
        freqs, psd = sp_signal.welch(epoch[:, ch], fs, nperseg=nperseg)
        psds.append((freqs, psd))
        for band in freq_bands.values():
            feats.append(_bandpower(psd, freqs, band))

    # C3–C4 lateralisation: mu diff + beta diff  (2 features)
    f_c3, p_c3 = psds[2]
    f_c4, p_c4 = psds[4]
    c3_mu   = _bandpower(p_c3, f_c3, freq_bands["mu"])
    c4_mu   = _bandpower(p_c4, f_c4, freq_bands["mu"])
    c3_beta = _bandpower(p_c3, f_c3, freq_bands["beta"])
    c4_beta = _bandpower(p_c4, f_c4, freq_bands["beta"])
    feats += [c3_mu - c4_mu, c3_beta - c4_beta]

    # Motor / frontal ratio  (2 features)
    f_f3, p_f3 = psds[0]
    frontal = (_bandpower(p_f3, f_f3, freq_bands["mu"]) +
               _bandpower(p_f3, f_f3, freq_bands["beta"]))
    feats += [c3_mu / (frontal + 1e-10), c3_beta / (frontal + 1e-10)]

    # Time-domain std: C3, Cz, C4  (3 features)
    for ch_idx in [2, 3, 4]:
        feats.append(float(np.std(epoch[:, ch_idx])))

    return np.array(feats, dtype=np.float64)   # shape (28,)


def compute_erd_feature(
    epoch: np.ndarray,
    fs: float,
    freq_bands: dict,
) -> float:
    """
    ERD as a single scalar (dB).

    Splits the 3-second epoch into:
      first 1 s → pseudo-baseline power
      remaining 2 s → activation power

    At C3 and C4, computes mu + beta power in each half.
    Returns 10·log10(activation / baseline).

    Interpretation:
      Negative → desynchronisation → arm is moving
      Near zero → no power change → arm is at rest
    """
    n_base = int(fs * ERD_BASELINE_SECS)
    if epoch.shape[0] < n_base + int(fs * 0.5):
        return 0.0

    base_seg = epoch[:n_base, :]
    act_seg  = epoch[n_base:, :]

    def _mu_beta(seg: np.ndarray, ch: int) -> float:
        nperseg = min(256, seg.shape[0])
        f, p = sp_signal.welch(seg[:, ch], fs, nperseg=nperseg)
        return _bandpower(p, f, freq_bands["mu"]) + _bandpower(p, f, freq_bands["beta"])

    base_power = np.mean([_mu_beta(base_seg, 2),   # C3
                          _mu_beta(base_seg, 4)])  # C4
    act_power  = np.mean([_mu_beta(act_seg,  2),
                          _mu_beta(act_seg,  4)])

    # Guard both numerator and denominator so log10 never sees 0.
    # When signal is flat (no electrodes), both powers ≈ 0 and ratio → 1 → 0 dB.
    ratio = (act_power + 1e-10) / (base_power + 1e-10)
    return float(10.0 * np.log10(ratio))


def build_feature_vector(
    epoch: np.ndarray,
    cycle_pos: int,
    fs: float,
    freq_bands: dict,
) -> np.ndarray:
    """
    Full 30-feature vector for the 3-class model:
      [0:28]  frequency features (28)
      [28]    ERD_dB              (1)
      [29]    cycle_pos / 3.0     (1)
    """
    return np.concatenate([
        extract_freq_features(epoch, fs, freq_bands),
        [compute_erd_feature(epoch, fs, freq_bands)],
        [cycle_pos / 3.0],
    ])


def _maybe_resample(data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    if abs(orig_fs - target_fs) < 1.0:
        return data
    n = int(round(data.shape[0] * target_fs / orig_fs))
    return sp_signal.resample(data, n, axis=0)


# ── 3-class XGBoost wrapper ───────────────────────────────────────────────────

class ThreeClassClassifier:
    """Wraps the model artefacts saved by train_3class_model.ipynb."""

    def __init__(self, model_dir: str) -> None:
        paths = {
            "model":  os.path.join(model_dir, "xgb_3class_model.pkl"),
            "scaler": os.path.join(model_dir, "3class_scaler.pkl"),
            "le":     os.path.join(model_dir, "3class_label_encoder.pkl"),
            "meta":   os.path.join(model_dir, "3class_metadata.json"),
        }
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            sys.exit(
                "Missing model files:\n" +
                "\n".join(f"  {p}" for p in missing) +
                "\nRun  train_3class_model.ipynb  first, then restart this script."
            )

        with open(paths["model"],  "rb") as f: self.model  = pickle.load(f)
        with open(paths["scaler"], "rb") as f: self.scaler = pickle.load(f)
        with open(paths["le"],     "rb") as f: self.le     = pickle.load(f)
        with open(paths["meta"])         as f: self.meta   = json.load(f)

        self.train_fs   = float(self.meta["training_sampling_rate"])
        self.bandpass   = tuple(self.meta["bandpass_hz"])
        self.notch_hz   = float(self.meta["notch_hz"])
        self.freq_bands = {k: tuple(v) for k, v in self.meta["freq_bands"].items()}
        self.classes    = self.meta["classes"]

        algo = self.meta.get("algorithm", "XGBoost")
        print(
            f"{algo} 3-class model loaded\n"
            f"  Classes     : {self.classes}\n"
            f"  Test acc    : {self.meta['test_accuracy'] * 100:.1f}%  "
            f"(chance = {100/3:.1f}%)\n"
            f"  CV acc      : {self.meta['cv_accuracy_mean'] * 100:.1f}%"
            f" ± {self.meta['cv_accuracy_std'] * 100:.1f}%\n"
            f"  Features    : {self.meta['n_features']}  "
            f"({self.meta.get('features_description', '28 freq + ERD + cycle_pos')})\n"
            f"  Epoch       : {self.meta['ml_epoch_seconds']:.1f} s after PostureStart\n"
        )

    def classify(
        self,
        raw_epoch: np.ndarray,
        baseline_mean: np.ndarray,
        cycle_pos: int,
        live_fs: float,
    ) -> tuple[str, float, np.ndarray, float]:
        """
        Classify a raw EEG epoch.

        Parameters
        ----------
        raw_epoch      : (N_samples × 7) raw EEG in µV
        baseline_mean  : (7,) mean of the pre-trial baseline window
        cycle_pos      : 0–3, position in the repeating 4-trial block
        live_fs        : sampling rate of the incoming data (Hz)

        Returns
        -------
        label   : "NoMovement", "Flexing", "Extending", or "ARTIFACT"
        prob    : confidence of the predicted class (0–1)
        proba   : probability for each class in self.classes order
        erd_val : ERD_dB feature value (feature index 28)
        """
        # Resample to training rate if necessary
        ep = _maybe_resample(raw_epoch, live_fs, self.train_fs)

        # Baseline correction
        ep = ep - baseline_mean

        # Artifact check on motor channels (C3, Cz, C4)
        ptp = float(np.max(np.ptp(ep[:, [2, 3, 4]], axis=0)))
        if ptp > ARTIFACT_THRESHOLD:
            return "ARTIFACT", 0.0, np.array([]), float("nan")

        # Bandpass + notch filter (must match training)
        ep = _bandpass(ep, self.bandpass[0], self.bandpass[1], self.train_fs)
        ep = _notch(ep, self.notch_hz, self.train_fs)

        # Build 30-feature vector
        feat    = build_feature_vector(ep, cycle_pos, self.train_fs, self.freq_bands)
        erd_val = float(feat[28])   # ERD_dB is at index 28

        # Replace any remaining inf/nan (e.g. flat signal edge cases) with 0
        if not np.all(np.isfinite(feat)):
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        feat_s  = self.scaler.transform(feat[np.newaxis, :])

        proba    = self.model.predict_proba(feat_s)[0]
        pred_idx = int(np.argmax(proba))
        label    = self.le.inverse_transform([pred_idx])[0]
        prob     = float(proba[pred_idx])

        return label, prob, proba, erd_val


# ── LSL helpers ───────────────────────────────────────────────────────────────

def find_xon_stream(timeout: float):
    streams = resolve_streams(wait_time=timeout)
    for s in streams:
        if s.type() == "EEG" and s.name().startswith("X.on"):
            return s
    return None


def find_marker_stream(timeout: float):
    streams = resolve_streams(wait_time=timeout)
    for s in streams:
        if s.name() == MARKER_STREAM_NAME:
            return s
    return None


def open_serial(port: str, baud: int):
    if not port:
        return None
    try:
        import serial
        ser = serial.Serial(port, baudrate=baud, timeout=0)
        print(f"Serial port opened: {port} @ {baud} baud")
        return ser
    except Exception as e:
        print(f"Serial open failed ({e}). Triggers will be printed to stdout.")
        return None


def send_trigger(ser, message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    if ser is None:
        print(f"    [{ts}] TRIGGER → {message}")
    else:
        ser.write((message + "\n").encode("ascii"))
        print(f"    [{ts}] Sent via serial: {message}")


def parse_trial_number(marker: str) -> Optional[int]:
    """Extract N from 'PostureStart_Trial_N'. Returns None otherwise."""
    m = re.search(r"PostureStart_Trial_(\d+)", marker)
    return int(m.group(1)) if m else None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-class BCI: NoMovement / Flexing / Extending.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Directory containing the four .pkl / .json model files",
    )
    parser.add_argument(
        "--serial-port", default="",
        help="Serial port for exoskeleton trigger (e.g. COM3). "
             "Leave empty to print triggers to stdout.",
    )
    parser.add_argument("--baud",           type=int,   default=115200)
    parser.add_argument(
        "--erd-threshold", type=float, default=-1.0,
        help="ERD gate threshold in dB. "
             "Epochs with ERD > this value are classified as NoMovement without "
             "consulting the ML model (no desynchronisation = no arm movement). "
             "More negative = stricter gate (requires stronger desync to trigger ML). "
             "Typical range: -0.5 to -2.5 dB. Default: -1.0 dB.",
    )
    parser.add_argument("--eeg-timeout",    type=float, default=10.0,
                        help="Seconds to wait for X.on EEG stream")
    parser.add_argument("--marker-timeout", type=float, default=10.0,
                        help="Seconds to wait for XonMarkerStream")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    classifier = ThreeClassClassifier(args.model_dir)

    # ── Connect EEG stream ────────────────────────────────────────────────────
    print("Looking for X.on EEG stream...")
    eeg_info = find_xon_stream(args.eeg_timeout)
    if eeg_info is None:
        sys.exit(
            "No X.on EEG stream found.\n"
            "Start the X.on app, set 500 Hz, click Link, then restart this script."
        )

    live_fs   = float(eeg_info.nominal_srate() or 500.0)
    eeg_inlet = StreamInlet(eeg_info)
    print(f"EEG connected : {eeg_info.name()}  fs={live_fs:.0f} Hz")

    # ── Connect marker stream ─────────────────────────────────────────────────
    print(f"Looking for {MARKER_STREAM_NAME}...")
    mrk_info = find_marker_stream(args.marker_timeout)
    if mrk_info is None:
        sys.exit(
            f"'{MARKER_STREAM_NAME}' not found.\n"
            "Start experiment_calibration.py in another terminal first, then restart."
        )
    mrk_inlet = StreamInlet(mrk_info)
    print(f"Markers       : {mrk_info.name()} connected\n")

    # ── Rolling EEG buffer ────────────────────────────────────────────────────
    buf_len              = int(BUFFER_SECS * live_fs)
    eeg_buf: deque       = deque(maxlen=buf_len)   # each element: (7,) array
    ts_buf:  deque       = deque(maxlen=buf_len)   # each element: float LSL timestamp

    # ── Running baseline accumulator (updated each trial) ─────────────────────
    baseline_sum   = np.zeros(N_CHANNELS)
    baseline_count = 0

    # ── Serial port ───────────────────────────────────────────────────────────
    ser = open_serial(args.serial_port, args.baud)

    # ── Per-trial state ───────────────────────────────────────────────────────
    # Set when a PostureStart marker is received:
    pending_lsl_ts:   Optional[float] = None   # LSL timestamp of PostureStart
    pending_mono_t:   Optional[float] = None   # monotonic time when marker received
    pending_trial_n:  Optional[int]   = None   # trial number from marker label
    pending_cycle_pos: Optional[int]  = None   # (trial_n - 1) % 4

    total_trials  = 0
    total_preds   = 0

    print("=" * 65)
    print(" 3-CLASS BCI  —  NoMovement / Flexing / Extending")
    print(f" Stage 1 ERD  : if ERD > {args.erd_threshold:+.1f} dB  → NoMovement (no ML call)")
    print(f" Stage 2 ML   : if ERD ≤ {args.erd_threshold:+.1f} dB  → XGBoost (Flexing / Extending)")
    print(f" Trigger      : PostureStart_Trial_N markers ({MARKER_STREAM_NAME})")
    print(f" Epoch        : {EPOCH_SECS:.1f} s after each PostureStart")
    print(f" Serial out   : {'yes (' + args.serial_port + ')' if args.serial_port else 'no (stdout only)'}")
    print(f" Tune gate    : --erd-threshold <dB>  (more negative = stricter)")
    print(" Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    pull_t    = 0.02          # 20 ms pull loop — tight enough for marker latency
    max_samps = max(1, int(live_fs * pull_t))

    # ── Pre-fill EEG buffer so the baseline window is available on trial 1 ───
    warmup_samples = int(BASELINE_PRE_SECS * live_fs)   # 5 s = 2500 samples
    print(f"  Pre-filling {BASELINE_PRE_SECS:.0f} s EEG buffer before accepting markers...",
          end=" ", flush=True)
    while len(eeg_buf) < warmup_samples:
        chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1, max_samples=max_samps)
        for samp, ts in zip(chunk, timestamps):
            eeg_buf.append(np.array(samp[:N_CHANNELS], dtype=np.float64))
            ts_buf.append(float(ts))
        # Drain any queued markers so they don't pile up during warmup
        mrk_inlet.pull_sample(timeout=0.0)
    print("ready.\n")

    try:
        while True:
            # ── 1. Drain EEG into rolling buffer ──────────────────────────────
            chunk, timestamps = eeg_inlet.pull_chunk(
                timeout=pull_t, max_samples=max_samps
            )
            for samp, ts in zip(chunk, timestamps):
                eeg_buf.append(np.array(samp[:N_CHANNELS], dtype=np.float64))
                ts_buf.append(float(ts))

            # ── 2. Check for new marker (non-blocking) ────────────────────────
            mrk_samp, mrk_lsl_ts = mrk_inlet.pull_sample(timeout=0.0)
            if mrk_samp and pending_lsl_ts is None:
                marker_str = str(mrk_samp[0])
                trial_n    = parse_trial_number(marker_str)
                if trial_n is not None:
                    cycle_pos = (trial_n - 1) % 4
                    total_trials += 1
                    pending_lsl_ts    = float(mrk_lsl_ts)
                    pending_mono_t    = time.monotonic()
                    pending_trial_n   = trial_n
                    pending_cycle_pos = cycle_pos
                    print(
                        f"\n  Trial {trial_n:3d}  |  cycle_pos={cycle_pos}"
                        f"  |  Collecting {EPOCH_SECS:.1f} s epoch..."
                    )

            # ── 3. When epoch is complete, classify ───────────────────────────
            if (pending_mono_t is not None and
                    time.monotonic() - pending_mono_t >= EPOCH_SECS):

                trial_n   = pending_trial_n
                cycle_pos = pending_cycle_pos
                evt_ts    = pending_lsl_ts

                # Clear state immediately so the next marker can be accepted
                pending_lsl_ts    = None
                pending_mono_t    = None
                pending_trial_n   = None
                pending_cycle_pos = None

                # Convert rolling buffers to arrays
                ts_arr  = np.array(ts_buf,  dtype=np.float64)
                eeg_arr = np.array(eeg_buf, dtype=np.float64)

                # Epoch: from PostureStart for EPOCH_SECS
                epoch_mask = (ts_arr >= evt_ts) & (ts_arr < evt_ts + EPOCH_SECS)
                raw_epoch  = eeg_arr[epoch_mask]

                # Baseline: 1-second window from −5 s to −4 s
                bsl_mask = (
                    (ts_arr >= evt_ts - BASELINE_PRE_SECS) &
                    (ts_arr <  evt_ts - BASELINE_POST_SECS)
                )
                raw_bsl = eeg_arr[bsl_mask]

                # Compute baseline mean (with fallbacks)
                if raw_bsl.shape[0] > 10:
                    bsl_mean = np.mean(raw_bsl, axis=0)
                    baseline_sum   += bsl_mean
                    baseline_count += 1
                elif baseline_count > 0:
                    bsl_mean = baseline_sum / baseline_count
                    print("    (Baseline window unavailable — using running mean)")
                else:
                    bsl_mean = np.zeros(N_CHANNELS)
                    print("    (No baseline available yet — using zeros)")

                # Sanity-check epoch length
                min_samples = int(live_fs * EPOCH_SECS * 0.7)
                if raw_epoch.shape[0] < min_samples:
                    print(
                        f"  Trial {trial_n:3d}: SKIPPED  "
                        f"only {raw_epoch.shape[0]} samples in buffer "
                        f"(need ≥ {min_samples}).  "
                        f"Is the EEG buffer large enough?"
                    )
                    continue

                # ── Two-stage classification ──────────────────────────────
                try:
                    ts_str = time.strftime("%H:%M:%S")

                    # Stage 1: compute ERD from the raw epoch (before full ML)
                    # Preprocess just enough to get ERD
                    ep_tmp = _maybe_resample(raw_epoch, live_fs, classifier.train_fs)
                    ep_tmp = ep_tmp - bsl_mean
                    ep_tmp = _bandpass(ep_tmp,
                                       classifier.bandpass[0],
                                       classifier.bandpass[1],
                                       classifier.train_fs)
                    ep_tmp = _notch(ep_tmp, classifier.notch_hz, classifier.train_fs)
                    erd_val = compute_erd_feature(ep_tmp, classifier.train_fs,
                                                  classifier.freq_bands)
                    erd_str = f"{erd_val:+.2f}" if not np.isnan(erd_val) else "N/A"

                    # Artifact check (peak-to-peak at motor channels)
                    ptp = float(np.max(np.ptp(ep_tmp[:, [2, 3, 4]], axis=0)))
                    if ptp > ARTIFACT_THRESHOLD:
                        total_preds += 1
                        print(
                            f"  [{ts_str}] Trial {trial_n:3d}: "
                            f"ARTIFACT — epoch too noisy (ptp={ptp:.0f} µV), skipped"
                        )
                        continue

                    if np.isnan(erd_val) or erd_val > args.erd_threshold:
                        # ── Stage 1 says: no movement ─────────────────────────
                        total_preds += 1
                        print(
                            f"  [{ts_str}] Trial {trial_n:3d}: "
                            f"→ NoMovement    [ERD gate: {erd_str} dB "
                            f"> threshold {args.erd_threshold:+.1f} dB]"
                        )
                        # No trigger

                    else:
                        # ── Stage 2: ERD says movement — ask ML ───────────────
                        # Trust whatever ML returns; it has the full feature picture.
                        label, prob, proba, _ = classifier.classify(
                            raw_epoch, bsl_mean, cycle_pos, live_fs
                        )
                        total_preds += 1

                        if label == "ARTIFACT":
                            print(
                                f"  [{ts_str}] Trial {trial_n:3d}: "
                                f"ARTIFACT — epoch too noisy, skipped"
                            )
                        else:
                            prob_parts = [
                                f"{cls}={p:.3f}"
                                for cls, p in zip(classifier.classes, proba)
                            ]
                            print(
                                f"  [{ts_str}] Trial {trial_n:3d}: "
                                f"→ {label:<12s}  prob={prob:.3f}  "
                                f"ERD={erd_str} dB  "
                                f"[{',  '.join(prob_parts)}]"
                            )
                            # Send trigger only for movement; if ML still says
                            # NoMovement despite ERD gate passing, trust the ML.
                            if label in ("Flexing", "Extending"):
                                send_trigger(ser, f"ML_{label.upper()}")
                            else:
                                print("    (ML voted NoMovement despite ERD gate — no trigger)")

                except Exception as exc:
                    print(f"  Trial {trial_n:3d}: Classification error — {exc}")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        if ser is not None:
            ser.close()

    print(f"\nSession summary:")
    print(f"  Trials seen       : {total_trials}")
    print(f"  Predictions made  : {total_preds}")
    if total_trials > total_preds:
        print(f"  Skipped / artifact: {total_trials - total_preds}")


if __name__ == "__main__":
    main()
