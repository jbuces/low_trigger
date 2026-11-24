"""
Stereo trigger of the mono trigger chains: multiplicity and topological (FEB coincidences).
"""
from math import comb

import numpy as np
import pandas as pd
from astropy.table import Table

N_FEB = 163


def load_feb_counts(path):
    """{(t1, t2): counts (163, 163)} from the output of scripts/make_feb_coincidences.py."""
    data = np.load(path)
    return {tuple(int(t) for t in key.split("_")): data[key] for key in data.files}


def feb_lut(counts, threshold=0.05):
    """
    Look-up table of allowed FEB coincidences, {(t1, t2): allowed (163, 163)}: allowed[f1, f2] is True
    if FEB f2 of t2 holds more than `threshold` of the showers with their maximum in FEB f1 of t1.
    """
    lut = {}
    for pair, c in counts.items():
        total = c.sum(axis=1, keepdims=True)
        frac = np.divide(c, total, out=np.zeros(c.shape), where=total > 0)
        lut[pair] = frac > threshold
    return lut


def mean_coincident_febs(lut):
    """Mean number of allowed FEBs per FEB (empty FEBs count as 0), per telescope pair and overall."""
    per_pair = {pair: float(allowed.sum(axis=1).mean()) for pair, allowed in lut.items()}
    return per_pair, float(np.mean(list(per_pair.values())))


def stereo_nsb_rate(mono_rate_hz, n_tels=4, min_tels=2, window_ns=20.0, topo_factor=1.0):
    """
    Accidental stereo rate of NSB triggers among `n_tels` independent telescopes with mono rate R:
    C(n, k) * k * R^k * tau^(k-1), with tau the coincidence half window, times topo_factor^(k-1)
    for the topological condition (topo_factor = mean coincident FEBs / 163).
    """
    k, tau = min_tels, window_ns * 1e-9
    return comb(n_tels, k) * k * mono_rate_hz ** k * tau ** (k - 1) * topo_factor ** (k - 1)


class StereoChain:
    """
    Stereo trigger built on a mono TriggerChain. An event is triggered if at least `min_tels` telescopes
    trigger with the mono chain and, with a look-up table, if at least `min_tels` of them take part in an
    allowed FEB coincidence. Its gamma table has one row per event, so the efficiency must be computed
    with SimulatedShowers(..., n_tel=1).
    """
    has_cnn = False

    def __init__(self, mono, min_tels=2, lut=None, topo_factor=None, n_tels=4, window_ns=20.0, name=None):
        self.mono, self.min_tels, self.lut = mono, min_tels, lut
        if lut is not None and topo_factor is None:
            topo_factor = mean_coincident_febs(lut)[1] / N_FEB        # from the look-up table itself
        self.topo_factor, self.n_tels, self.window_ns = topo_factor, n_tels, window_ns
        self.name = name or f"{mono.name} + stereo {min_tels}{' topo' if lut is not None else ''}"

        g = mono.gamma
        df = pd.DataFrame({"obs_id": np.asarray(g["obs_id"]), "event_id": np.asarray(g["event_id"]),
                           "tel_id": np.asarray(g["tel_id"]), "true_energy": np.asarray(g["true_energy"]),
                           "trig": mono.gamma_triggered()})
        events = df.groupby(["obs_id", "event_id"], sort=True)
        table = events.agg(true_energy=("true_energy", "first"), n_trig=("trig", "sum")).reset_index()
        triggered = (table["n_trig"] >= min_tels).to_numpy()

        if lut is not None:
            if hasattr(mono, "triggered_febs"):                       # ScanChain: FEBs for the current threshold
                febs = mono.triggered_febs()
            elif "triggered_febs" in g.colnames:                      # trigger files
                febs = np.asarray(g["triggered_febs"]).astype(bool)
            else:
                raise ValueError("The topological trigger needs the 'triggered_febs' column in the trigger files.")
            trig_pos = np.flatnonzero(df["trig"].to_numpy())               # positions of the triggered frames
            frames = df.iloc[trig_pos].groupby(["obs_id", "event_id"]).indices
            for row in np.flatnonzero(triggered):
                key = (table["obs_id"].iat[row], table["event_id"].iat[row])
                idx = trig_pos[frames[key]]                                 # back to positions in the full table
                tels = df["tel_id"].to_numpy()[idx]
                involved = set()
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        allowed = lut[(int(tels[a]), int(tels[b]))]
                        if allowed[np.ix_(febs[idx[a]], febs[idx[b]])].any():
                            involved.update((tels[a], tels[b]))
                triggered[row] = len(involved) >= min_tels

        self.gamma = Table({"obs_id": table["obs_id"].to_numpy(), "event_id": table["event_id"].to_numpy(),
                            "tel_id": np.zeros(len(table), dtype=int), "true_energy": table["true_energy"].to_numpy()})
        self._triggered = triggered

    def gamma_triggered(self):
        return self._triggered

    def output_rate(self):
        """Accidental stereo NSB rate (Hz) from the mono output rate."""
        return stereo_nsb_rate(self.mono.output_rate(), self.n_tels, self.min_tels, self.window_ns,
                               self.topo_factor if self.lut is not None else 1.0)

def lut_summary(counts, thresholds=(0.0, 0.00065, 0.01, 0.05)):
    """Mean number of coincident FEBs and topological factor (mean / 163) per threshold and telescope pair."""
    pairs = sorted(counts)
    print(f"{'threshold':>10} " + " ".join(f"{f'{a}-{b}':>7}" for a, b in pairs) + f" {'mean':>7} {'factor':>8}")
    summary = {}
    for thr in thresholds:
        per_pair, mean = mean_coincident_febs(feb_lut(counts, thr))
        summary[thr] = (per_pair, mean)
        print(f"{thr:10.5f} " + " ".join(f"{per_pair[p]:7.2f}" for p in pairs) + f" {mean:7.2f} {mean / N_FEB:8.4f}")
    return summary


def save_feb_lut(lut, path):
    """Save a look-up table (one allowed matrix (163, 163) per telescope pair) as a .npz file."""
    np.savez(path, **{f"{t1}_{t2}": allowed for (t1, t2), allowed in lut.items()})

def stereo_rate_vs_threshold(mono, sims_evt, bins, thresholds, min_tels=2, lut=None, **cosmic_kwargs):
    """
    Stereo NSB rate and cosmic-ray rate (Hz, per array event) for a range of thresholds of the mono chain:
    for each threshold the stereo chain (multiplicity, or topological with ``lut``) is rebuilt.
    ``sims_evt`` must use n_tel=1. ``cosmic_kwargs`` are passed to ``cosmic_ray_rate``.
    """
    from .efficiency import cosmic_ray_rate
    saved = mono.threshold
    nsb, cosmic = [], []
    for thr in thresholds:
        mono.threshold = thr
        st = StereoChain(mono, min_tels=min_tels, lut=lut)
        nsb.append(st.output_rate())
        cosmic.append(cosmic_ray_rate(st, sims_evt, bins, **cosmic_kwargs))
    mono.threshold = saved
    return np.asarray(thresholds, dtype=float), np.array(nsb), np.array(cosmic)