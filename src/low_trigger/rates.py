"""
Trigger rates on NSB-only frames.

All functions take the ``trigger_per_sample`` column of the trigger files: an array
with shape (n_frames, n_samples) of 0/1, one row per telescope frame.
"""
import numpy as np


def _as_bool(trigger_per_sample, edge=0):
    """
    0/1 array as booleans, dropping ``edge`` samples at both ends of each frame
    (e.g. edge=1 for TDSCAN with eps_t=1, whose first and last samples use a truncated window).
    """
    tps = np.asarray(trigger_per_sample).astype(bool)
    if edge:
        tps = tps[:, edge:tps.shape[1] - edge]
    return tps


def frame_rate(trigger_per_sample, dt_ns=1.0, edge=0):
    """Rate of frames with at least one trigger, P / T. Saturates at 1 / T."""
    tps = _as_bool(trigger_per_sample, edge=edge)
    frame_s = tps.shape[1] * dt_ns * 1e-9
    return tps.any(axis=1).mean() / frame_s


def poisson_rate(trigger_per_sample, dt_ns=1.0, edge=0):
    """Trigger rate assuming independent triggers, -ln(1 - P) / T."""
    tps = _as_bool(trigger_per_sample, edge=edge)
    frame_s = tps.shape[1] * dt_ns * 1e-9
    return -np.log1p(-tps.any(axis=1).mean()) / frame_s


def trigger_onsets(trigger_per_sample, edge=0):
    """
    Trigger onsets (0 -> 1) in the stream of all frames one after another.

    Returns the sample positions of the onsets and the total number of samples.
    """
    stream = _as_bool(trigger_per_sample, edge=edge).ravel()
    onsets = np.flatnonzero(stream[1:] & ~stream[:-1]) + 1
    return onsets, stream.size


def trigger_rate(trigger_per_sample, dt_ns=1.0, edge=0):
    """Number of trigger onsets per second."""
    onsets, n_samples = trigger_onsets(trigger_per_sample, edge=edge)
    return len(onsets) / (n_samples * dt_ns * 1e-9)


def window_stats(trigger_per_sample, window=10, dt_ns=1.0, edge=0):
    """
    Windows opened by the triggers in the stream.

    Each trigger at sample t opens the window [t - 1, t + window - 2]. A trigger that
    arrives while a window is open does not open a new one.

    Returns a dict with:
    - ``trigger_rate``: onsets per second (CNN load if every trigger is evaluated)
    - ``window_rate``: non overlapping windows per second
    - ``frac_second_trigger``: fraction of windows with another trigger inside
    - ``busy_fraction``: fraction of the time with an open window
    """
    onsets, n_samples = trigger_onsets(trigger_per_sample, edge=edge)
    total_s = n_samples * dt_ns * 1e-9

    starts = []
    free_from = 0
    for t in onsets:
        if t - 1 >= free_from:
            starts.append(t)
            free_from = t - 1 + window
    starts = np.array(starts)

    ends = starts - 1 + window
    n_inside = np.searchsorted(onsets, ends, side="left") - np.searchsorted(onsets, starts, side="right")

    return {
        "trigger_rate": len(onsets) / total_s,
        "window_rate": len(starts) / total_s,
        "frac_second_trigger": (n_inside > 0).mean() if len(starts) else 0.0,
        "busy_fraction": len(starts) * window / n_samples,
    }
