from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_streams

from erd_calculator import ERDCalculator


XON_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]


def find_xon_stream(timeout: float):
    streams = resolve_streams(wait_time=timeout)
    for stream in streams:
        if stream.type() == "EEG" and stream.name().startswith("X.on"):
            return stream
    raise RuntimeError("No X.on EEG LSL stream found.")


def parse_channels(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_focus_indices(raw: str) -> list[int]:
    return [int(item.strip()) - 1 for item in raw.split(",") if item.strip()]


def choose_focus_for_arm(arm: str, channel_names: list[str]) -> list[int]:
    if arm == "right" and "C3" in channel_names:
        return [channel_names.index("C3")]
    if arm == "left" and "C4" in channel_names:
        return [channel_names.index("C4")]
    if "Cz" in channel_names:
        return [channel_names.index("Cz")]
    return [0]


def open_serial(port: str, baud: int):
    if not port:
        return None
    import serial

    return serial.Serial(port, baudrate=baud, timeout=0)


def send_trigger(serial_port, message: str) -> None:
    if serial_port is None:
        print(f"\nTRIGGER {time.strftime('%H:%M:%S')} {message}")
        return
    serial_port.write((message + "\n").encode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ERD detector for X.ON arm movement.")
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--method", choices=["bandpass", "welch", "db_correction"], default="db_correction")
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--activation-seconds", type=float, default=1.0)
    parser.add_argument("--bandpass-low", type=float, default=8.0)
    parser.add_argument("--bandpass-high", type=float, default=30.0)
    parser.add_argument("--xon-channels", default="F3,F4,C3,Cz,C4,P3,P4")
    parser.add_argument(
        "--focus-channels",
        default="",
        help="Optional 1-based channel indices to use, e.g. 3 or 3,4,5. Default picks C3 for right arm, C4 for left arm.",
    )
    parser.add_argument(
        "--erd-threshold",
        type=float,
        default=-1.5,
        help="Trigger when mean ERD is at or below this value. Negative values indicate desynchronization.",
    )
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--refractory", type=float, default=2.0)
    parser.add_argument("--update-hz", type=float, default=5.0)
    parser.add_argument("--ptp-thresh", type=float, default=1500.0)
    parser.add_argument("--serial-port", default="")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--message", default="ARM_MOVE")
    args = parser.parse_args()

    channel_names = parse_channels(args.xon_channels)
    focus_indices = (
        parse_focus_indices(args.focus_channels)
        if args.focus_channels
        else choose_focus_for_arm(args.arm, channel_names)
    )
    focus_names = [channel_names[idx] for idx in focus_indices]

    eeg_info = find_xon_stream(timeout=5.0)
    fs = float(eeg_info.nominal_srate() or 250.0)
    inlet = StreamInlet(eeg_info)
    serial_port = open_serial(args.serial_port, args.baud)

    calculator = ERDCalculator(
        sampling_freq=fs,
        epoch_pre_stimulus_seconds=args.baseline_seconds,
        epoch_post_stimulus_seconds=args.activation_seconds,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        channel_names=channel_names,
        focus_channels_indices=focus_indices,
        focus_stimuli=None,
        apply_car=len(focus_indices) > 1,
    )

    buffer_samples = calculator.epoch_total_samples
    buffer = deque(maxlen=buffer_samples)
    positive_streak = 0
    last_trigger = 0.0

    print(f"Listening to {eeg_info.name()} at {fs:.2f} Hz.")
    print(
        f"Method={args.method}, baseline={args.baseline_seconds:.2f}s, activation={args.activation_seconds:.2f}s, "
        f"focus={focus_names}, threshold={args.erd_threshold:.2f}"
    )
    print("This detector is movement-sensitive only. It does not distinguish flexion from extension.")

    try:
        while True:
            max_samples = int(fs / args.update_hz) or 1
            chunk, _ = inlet.pull_chunk(timeout=1.0 / args.update_hz, max_samples=max_samples)
            for sample in chunk:
                buffer.append(sample[: len(channel_names)])

            if len(buffer) < buffer_samples:
                continue

            epoch = np.asarray(buffer, dtype=np.float64).T
            focus_epoch = epoch[focus_indices, :]
            if np.max(np.ptp(focus_epoch, axis=1)) > args.ptp_thresh:
                positive_streak = 0
                continue

            erd_value = calculator.calculate_live_erd(epoch, method=args.method, return_mean=True)
            if erd_value is None or np.isnan(erd_value):
                positive_streak = 0
                continue

            if erd_value <= args.erd_threshold:
                positive_streak += 1
            else:
                positive_streak = 0

            now = time.monotonic()
            if positive_streak >= args.consecutive and now - last_trigger >= args.refractory:
                send_trigger(serial_port, args.message)
                last_trigger = now
                positive_streak = 0

            print(f"ERD={erd_value:.3f} focus={','.join(focus_names)}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if serial_port is not None:
            serial_port.close()


if __name__ == "__main__":
    main()
