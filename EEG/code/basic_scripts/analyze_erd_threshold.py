from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from erd_calculator import ERDCalculator


def parse_channel_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_focus_indices(raw: str) -> list[int]:
    return [int(item.strip()) - 1 for item in raw.split(",") if item.strip()]


def trial_label(trial_num: int, pattern: str) -> str:
    idx = (trial_num - 1) % 4
    if pattern.upper() == "A":
        labels = ["flex", "rest_flexed", "extend", "rest_extended"]
    else:
        labels = ["extend", "rest_extended", "flex", "rest_flexed"]
    return labels[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ERD values from X.ON merged CSV files.")
    parser.add_argument("inputs", nargs="+", help="Merged CSV files")
    parser.add_argument("--pattern", choices=["A", "B"], required=True)
    parser.add_argument("--method", choices=["bandpass", "welch", "db_correction"], default="db_correction")
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--activation-seconds", type=float, default=1.0)
    parser.add_argument("--bandpass-low", type=float, default=8.0)
    parser.add_argument("--bandpass-high", type=float, default=30.0)
    parser.add_argument("--xon-channels", default="F3,F4,C3,Cz,C4,P3,P4")
    parser.add_argument("--focus-channels", default="3")
    parser.add_argument("--output", default="data/results/erd_summary.csv")
    args = parser.parse_args()

    channel_names = parse_channel_names(args.xon_channels)
    focus_indices = parse_focus_indices(args.focus_channels)
    calculator = ERDCalculator(
        sampling_freq=args.fs,
        epoch_pre_stimulus_seconds=args.baseline_seconds,
        epoch_post_stimulus_seconds=args.activation_seconds,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        channel_names=channel_names,
        focus_channels_indices=focus_indices,
        focus_stimuli=None,
        apply_car=len(focus_indices) > 1,
    )

    rows = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        df = pd.read_csv(path)
        if "markers" not in df.columns:
            continue
        timestamps = df.iloc[:, 0].to_numpy(dtype=np.float64)
        data = df.iloc[:, 1:-1].to_numpy(dtype=np.float64)
        markers = df["markers"].fillna("").astype(str).tolist()
        sort_idx = np.argsort(timestamps)
        data = data[sort_idx]
        markers = [markers[i] for i in sort_idx]

        event_rows = []
        for idx, marker in enumerate(markers):
            if marker.startswith("PostureStart_Trial_"):
                trial_num = int(marker.rsplit("_", 1)[-1])
                event_rows.append((idx, trial_num))

        for event_idx, trial_num in event_rows:
            start = event_idx - calculator.samples_before_marker
            end = event_idx + calculator.samples_after_marker
            if start < 0 or end > len(data):
                continue
            epoch = data[start:end].T
            if args.method == "bandpass":
                erd = calculator.calculate_erd_from_bandpass(epoch, return_mean=True)
            elif args.method == "welch":
                erd = calculator.calculate_erd_from_welch(epoch, return_mean=True)
            else:
                erd = calculator.calculate_erd_from_db_correction(epoch, return_mean=True)
            rows.append(
                {
                    "file": path.name,
                    "trial": trial_num,
                    "label": trial_label(trial_num, args.pattern),
                    "erd": erd,
                }
            )

    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved ERD summary: {output}")
    if result.empty:
        return

    print("\nMean ERD by label:")
    print(result.groupby("label")["erd"].agg(["count", "mean", "std", "min", "max"]))


if __name__ == "__main__":
    main()
