"""
Threshold scans of the L1 + OR directly from the R0 waveforms, without trigger files.

One pass over the waveforms stores, for every frame, the maximum over the camera of the flower sum
(Boolean OR) and of the FEB amplitude (Amplitude-Aware OR, for a given flower threshold). Any OR
threshold can then be applied to them:
- NSB: maximum per sample, shape (n_frames, n_samples), to compute the trigger rate;
- gamma rays: maximum in the shower window, one value per frame, to compute the efficiency.
"""
import numpy as np
import tables
from astropy.table import Table, vstack
from ctapipe.io import read_table

from . import rates

SHOWER_SAMPLES = (15, 25)     # shower window of the gamma rays (both included)


def load_flower_pixels(geometry_file):
    """Pixels of each flower, shape (1141, 7), from the AdvCam trigger geometry file."""
    with tables.open_file(geometry_file) as f:
        return f.root.neighbors.flower_neighbors_l1[:]


def _maxima(wf, flower_pixels, t_flower):
    """Per-sample maxima over the camera of the flower sum and of the FEB amplitude, each (n, T)."""
    fl = wf[:, flower_pixels].sum(axis=2)                                             # (n, 1141, T)
    amp = np.clip(fl - t_flower, 0, None).reshape(len(wf), 163, 7, -1).sum(axis=2)    # (n, 163, T)
    return fl.max(axis=1), amp.max(axis=1)


def scan_nsb(paths, flower_pixels, t_flower=2160, tel_id=1, chunk=500):
    """Per-sample maxima of the NSB frames: {'flower': (n, T), 'amplitude': (n, T)}."""
    flower, amplitude = [], []
    for path in paths:
        with tables.open_file(path) as h5:
            node = h5.get_node(f"/r0/event/telescope/tel_{tel_id:03d}")
            for start in range(0, node.nrows, chunk):
                wf = node.read(start, min(start + chunk, node.nrows), field="waveform")[:, 0].astype(np.int32)
                f, a = _maxima(wf, flower_pixels, t_flower)
                flower.append(f)
                amplitude.append(a)
    return {"flower": np.concatenate(flower), "amplitude": np.concatenate(amplitude)}


def scan_gamma(paths, flower_pixels, t_flower=2160, tel_ids=(1, 2, 3, 4), window=SHOWER_SAMPLES, chunk=500):
    """
    Maxima in the shower window of the gamma frames, with the columns needed for the efficiencies
    (obs_id, event_id, tel_id, true_energy, true_image_sum, max_flower, max_amplitude).
    """
    parts = []
    for path in paths:
        shower = read_table(path, "/simulation/event/subarray/shower")
        energy = dict(zip(zip(np.asarray(shower["obs_id"]), np.asarray(shower["event_id"])),
                          np.asarray(shower["true_energy"].to_value("TeV"))))
        with tables.open_file(path) as h5:
            for tel_id in tel_ids:
                name = f"/r0/event/telescope/tel_{tel_id:03d}"
                if name not in h5:
                    continue
                node = h5.get_node(name)
                images = h5.get_node(f"/simulation/event/telescope/images/tel_{tel_id:03d}")
                for start in range(0, node.nrows, chunk):
                    stop = min(start + chunk, node.nrows)
                    wf = node.read(start, stop, field="waveform")[:, 0, :, window[0]:window[1] + 1].astype(np.int32)
                    f, a = _maxima(wf, flower_pixels, t_flower)
                    obs = node.read(start, stop, field="obs_id")
                    evt = node.read(start, stop, field="event_id")
                    parts.append(Table({
                        "obs_id": obs, "event_id": evt, "tel_id": np.full(stop - start, tel_id),
                        "true_energy": [energy[(o, e)] for o, e in zip(obs, evt)],
                        "true_image_sum": images.read(start, stop, field="true_image_sum"),
                        "max_flower": f.max(axis=1), "max_amplitude": a.max(axis=1),
                    }))
    return vstack(parts)


def rates_vs_threshold(max_per_sample, thresholds):
    """Trigger rate (Hz) of each OR threshold, from the NSB per-sample maxima."""
    return np.array([rates.trigger_rate(max_per_sample >= thr) for thr in thresholds])


def threshold_for_rate(max_per_sample, thresholds, target_hz):
    """
    Lowest threshold whose rate does not exceed ``target_hz``, searched above the maximum of the
    rate curve (below it the OR is almost always on and the trigger rate is not meaningful).
    Returns the threshold and its rate (Hz).
    """
    r = rates_vs_threshold(max_per_sample, thresholds)
    peak = int(np.argmax(r))
    below = np.flatnonzero(r[peak:] <= target_hz)
    if not below.size:
        raise ValueError(f"No threshold reaches {target_hz / 1e6:.2f} MHz in the scanned range.")
    i = peak + below[0]
    return thresholds[i], r[i]


class ScanChain:
    """
    L1 + OR chain from the scans: a gamma frame triggers if its maximum in the shower window reaches
    ``threshold``. ``kind`` is 'flower' (Boolean OR) or 'amplitude' (Amplitude-Aware OR). It works with
    the efficiency and plotting functions like a TriggerChain.
    """
    has_cnn = False

    def __init__(self, name, gamma_scan, nsb_scan, kind, threshold):
        self.name, self.kind, self.threshold = name, kind, threshold
        self.gamma = gamma_scan
        self._nsb = nsb_scan[kind]

    def gamma_triggered(self):
        return np.asarray(self.gamma[f"max_{self.kind}"]) >= self.threshold

    def output_rate(self):
        return rates.trigger_rate(self._nsb >= self.threshold)