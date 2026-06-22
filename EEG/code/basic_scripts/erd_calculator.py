from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch


class ERDCalculator:
    """
    Compute event-related desynchronization (ERD) from EEG epochs.

    Epochs are expected as shape (n_channels, n_samples), with the first
    samples corresponding to the baseline segment and the second half to the
    activation segment.
    """

    def __init__(
        self,
        sampling_freq,
        epoch_pre_stimulus_seconds,
        epoch_post_stimulus_seconds,
        bandpass_low,
        bandpass_high,
        channel_names,
        focus_channels_indices,
        focus_stimuli=None,
        apply_car=True,
    ):
        self.sampling_freq = float(sampling_freq)
        self.samples_before_marker = int(round(epoch_pre_stimulus_seconds * sampling_freq))
        self.samples_after_marker = int(round(epoch_post_stimulus_seconds * sampling_freq))
        self.epoch_total_samples = self.samples_before_marker + self.samples_after_marker

        self.b, self.a = butter(
            5, [bandpass_low, bandpass_high], btype="band", fs=sampling_freq
        )

        self.channel_names = list(channel_names)
        self.channel_count = len(self.channel_names)
        self.focus_channels_indices = list(focus_channels_indices)
        self.bandpass_low = float(bandpass_low)
        self.bandpass_high = float(bandpass_high)
        self.focus_stimuli = focus_stimuli
        self.apply_car = bool(apply_car)

    def _preprocess_epoch(self, epoch_data):
        epoch = np.asarray(epoch_data, dtype=np.float64)
        if epoch.ndim != 2:
            return None
        if epoch.shape[0] != self.channel_count:
            return None
        if epoch.shape[1] < self.epoch_total_samples:
            return None

        filtered_epoch = filtfilt(self.b, self.a, epoch, axis=1)

        if self.apply_car and self.channel_count > 1:
            filtered_epoch = filtered_epoch - np.mean(filtered_epoch, axis=0, keepdims=True)

        return filtered_epoch

    def _compute_erd_percentage(self, pre_power, post_power):
        pre_power = np.asarray(pre_power, dtype=np.float64)
        post_power = np.asarray(post_power, dtype=np.float64)
        erd_percent = np.full(pre_power.shape, np.nan, dtype=np.float64)
        valid = pre_power != 0
        erd_percent[valid] = ((post_power[valid] - pre_power[valid]) / pre_power[valid]) * 100.0
        return erd_percent

    def _compute_erd_db(self, pre_power, post_power):
        pre_power = np.asarray(pre_power, dtype=np.float64)
        post_power = np.asarray(post_power, dtype=np.float64)
        erd_db = np.full(pre_power.shape, np.nan, dtype=np.float64)
        valid = (pre_power > 0) & (post_power > 0)
        erd_db[valid] = 10.0 * np.log10(post_power[valid] / pre_power[valid])
        return erd_db

    def _summarize_focus_channels(self, values, return_mean):
        values = np.asarray(values, dtype=np.float64)
        focus_values = values[self.focus_channels_indices]
        if return_mean:
            return float(np.nanmean(focus_values)) if np.any(~np.isnan(focus_values)) else None
        return {
            self.channel_names[i]: float(values[i]) if not np.isnan(values[i]) else np.nan
            for i in range(len(self.channel_names))
        }

    def calculate_erd_from_bandpass(self, epoch_data, return_mean=False):
        processed_epoch = self._preprocess_epoch(epoch_data)
        if processed_epoch is None:
            return None

        pre_stimulus_data = processed_epoch[:, : self.samples_before_marker]
        post_stimulus_data = processed_epoch[
            :, self.samples_before_marker : self.samples_before_marker + self.samples_after_marker
        ]

        pre_power = np.nanmean(pre_stimulus_data ** 2, axis=1)
        post_power = np.nanmean(post_stimulus_data ** 2, axis=1)
        erd_percent = self._compute_erd_percentage(pre_power, post_power)
        return self._summarize_focus_channels(erd_percent, return_mean)

    def calculate_erd_from_welch(self, epoch_data, return_mean=False):
        processed_epoch = self._preprocess_epoch(epoch_data)
        if processed_epoch is None:
            return None

        pre_stimulus_data = processed_epoch[:, : self.samples_before_marker]
        post_stimulus_data = processed_epoch[
            :, self.samples_before_marker : self.samples_before_marker + self.samples_after_marker
        ]

        f_pre, pxx_pre = welch(pre_stimulus_data, fs=self.sampling_freq, axis=1)
        f_post, pxx_post = welch(post_stimulus_data, fs=self.sampling_freq, axis=1)
        band_indices = (f_pre >= self.bandpass_low) & (f_pre <= self.bandpass_high)
        if not np.any(band_indices):
            return None

        pre_power = np.nanmean(pxx_pre[:, band_indices], axis=1)
        post_power = np.nanmean(pxx_post[:, band_indices], axis=1)
        erd_percent = self._compute_erd_percentage(pre_power, post_power)
        return self._summarize_focus_channels(erd_percent, return_mean)

    def calculate_erd_from_db_correction(self, epoch_data, return_mean=False):
        processed_epoch = self._preprocess_epoch(epoch_data)
        if processed_epoch is None:
            return None

        power_signal = processed_epoch ** 2
        pre_power = np.nanmean(power_signal[:, : self.samples_before_marker], axis=1)
        post_power = np.nanmean(
            power_signal[
                :, self.samples_before_marker : self.samples_before_marker + self.samples_after_marker
            ],
            axis=1,
        )
        erd_db = self._compute_erd_db(pre_power, post_power)
        return self._summarize_focus_channels(erd_db, return_mean)

    def calculate_erd_moving_average(
        self, epoch_data, window_size_samples, return_mean=True, method="percentage"
    ):
        processed_epoch = self._preprocess_epoch(epoch_data)
        if processed_epoch is None:
            return None

        pre_stimulus_data = processed_epoch[:, : self.samples_before_marker]
        post_stimulus_data = processed_epoch[
            :, self.samples_before_marker : self.samples_before_marker + self.samples_after_marker
        ]

        pre_len = pre_stimulus_data.shape[1]
        post_len = post_stimulus_data.shape[1]
        if not (0 < window_size_samples <= pre_len and window_size_samples <= post_len):
            return None

        num_windows = min(pre_len, post_len) - window_size_samples + 1
        if num_windows <= 0:
            return None

        all_window_erds = np.full((self.channel_count, num_windows), np.nan, dtype=np.float64)
        for i in range(num_windows):
            pre_power = np.mean(pre_stimulus_data[:, i : i + window_size_samples] ** 2, axis=1)
            post_power = np.mean(post_stimulus_data[:, i : i + window_size_samples] ** 2, axis=1)

            if method == "percentage":
                erd_window = self._compute_erd_percentage(pre_power, post_power)
            elif method in {"db", "db_correction"}:
                erd_window = self._compute_erd_db(pre_power, post_power)
            else:
                return None

            all_window_erds[:, i] = erd_window

        mean_erd_all_channels = np.nanmean(all_window_erds, axis=1)
        focus_window_trace = np.nanmean(all_window_erds[self.focus_channels_indices, :], axis=0)

        if return_mean:
            mean_focus = np.nanmean(mean_erd_all_channels[self.focus_channels_indices])
            return float(mean_focus), focus_window_trace
        return {
            self.channel_names[i]: float(mean_erd_all_channels[i])
            if not np.isnan(mean_erd_all_channels[i])
            else np.nan
            for i in range(len(self.channel_names))
        }

    def calculate_live_erd(self, epoch_data, method="db_correction", return_mean=True):
        """
        Continuous/live ERD calculation on a rolling buffer with
        baseline immediately followed by activation.
        """
        if method == "bandpass":
            return self.calculate_erd_from_bandpass(epoch_data, return_mean=return_mean)
        if method == "welch":
            return self.calculate_erd_from_welch(epoch_data, return_mean=return_mean)
        if method in {"db", "db_correction"}:
            return self.calculate_erd_from_db_correction(epoch_data, return_mean=return_mean)
        return None
