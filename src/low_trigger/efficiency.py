"""
Trigger efficiency and effective area of a trigger chain.
"""
import numpy as np
import pandas as pd


def energy_bins(e_min=5.0, e_max=5e4, width=0.1):
    """Logarithmic energy bins (GeV) with relative width ``width``, as in the previous plots."""
    n_edges = int(np.log10(e_max / e_min) / np.log10(1 + width)) + 1
    return np.logspace(np.log10(e_min), np.log10(e_max), n_edges)


def efficiency_vs_energy(chain, sims, bins, level="shower", reuse_factor=100):
    """
    Trigger efficiency per energy bin: fraction of the simulated telescope frames
    (showers x reuses x telescopes) triggered by the chain.

    The frames of the same shower are correlated, so the error is computed treating as
    independent units the showers (``level="shower"``) or the events of each reuse
    (``level="event"``).

    Returns the bin centers (GeV), the efficiency and its error.
    """
    table = chain.gamma
    obs_id = np.asarray(table["obs_id"])
    if sims.runs is not None and not np.isin(np.unique(obs_id), sims.runs).all():
        extra = np.setdiff1d(np.unique(obs_id), sims.runs).tolist()
        raise ValueError(f"'{chain.name}': runs {extra} are not in the simulated showers.")

    event_id = np.asarray(table["event_id"])
    if level == "shower":
        unit = event_id // reuse_factor
        trials_per_unit = sims.trials_per_shower        # reuse x n_tel
        n_units = sims.counts(bins).astype(float)       # showers per bin
    elif level == "event":
        unit = event_id
        trials_per_unit = sims.n_tel
        n_units = sims.counts(bins).astype(float) * sims.reuse
    else:
        raise ValueError("level must be 'shower' or 'event'.")

    # Triggered frames per unit: k triggers out of trials_per_unit possible
    triggered = chain.gamma_triggered()
    df = pd.DataFrame({
        "obs_id": obs_id[triggered],
        "unit": unit[triggered],
        "energy": np.asarray(table["true_energy"])[triggered] * 1e3,   # TeV -> GeV
    })
    per_unit = df.groupby(["obs_id", "unit"]).agg(k=("energy", "size"), energy=("energy", "first"))

    b = np.digitize(per_unit["energy"].values, bins) - 1
    ok = (b >= 0) & (b < len(bins) - 1)
    x = per_unit["k"].values[ok] / trials_per_unit       # triggered fraction of each unit
    b = b[ok]

    n_bins = len(bins) - 1
    sum_x = np.bincount(b, weights=x, minlength=n_bins)
    sum_x2 = np.bincount(b, weights=x**2, minlength=n_bins)
    # Units without any trigger count with x = 0 (they are included in n_units)
    eff = np.divide(sum_x, n_units, out=np.zeros(n_bins), where=n_units > 0)
    var = np.divide(sum_x2 - n_units * eff**2, n_units - 1, out=np.zeros(n_bins), where=n_units > 1)
    err = np.sqrt(np.clip(var, 0, None) / np.where(n_units > 0, n_units, 1))
    if np.any(eff > 1):
        print(f"WARNING '{chain.name}': efficiency > 1 in some bins. Do the runs of the gammas match the simulated showers?")

    centers = np.sqrt(bins[:-1] * bins[1:])
    return centers, eff, err


def efficiency_vs_pe(chain, bins):
    """
    Trigger efficiency per bin of true Cherenkov p.e. in the camera (``true_image_sum``).

    The denominator is all the gamma frames of the trigger files with those p.e., and the
    error is binomial (frames as independent trials).
    """
    pe = np.asarray(chain.gamma["true_image_sum"])
    triggered = chain.gamma_triggered()
    n, _ = np.histogram(pe, bins=bins)
    k, _ = np.histogram(pe[triggered], bins=bins)
    eff = np.divide(k, n, out=np.zeros(len(n)), where=n > 0)
    err = np.sqrt(np.divide(eff * (1 - eff), n, out=np.zeros(len(n)), where=n > 0))
    centers = np.sqrt(bins[:-1] * bins[1:])
    return centers, eff, err

def efficiency_vs_rate(chain, sims, energy_range=(10.0, 30.0), thresholds=None, n_points=50):
    """
    Trigger efficiency in an energy band (GeV) vs NSB output rate, moving the threshold of the chain:
    the CNN threshold for a TriggerChain with CNN, or the OR thresholds given in ``thresholds`` for a
    ScanChain. A chain without a threshold to move (TriggerChain without CNN) gives a single point.
    Returns the rates (Hz), efficiencies and errors.
    """
    band = np.asarray(energy_range, dtype=float)
    if getattr(chain, "has_cnn", False):
        p = np.asarray(chain.nsb["cnn_prediction"], dtype=float)
        thresholds = np.quantile(p[np.isfinite(p)], np.linspace(0.002, 1.0, n_points))
    elif thresholds is None:
        _, eff, err = efficiency_vs_energy(chain, sims, band)
        return np.array([chain.output_rate()]), eff, err

    saved = chain.threshold
    out_rates, effs, errs = [], [], []
    for thr in thresholds:
        chain.threshold = thr
        out_rates.append(chain.output_rate())
        _, eff, err = efficiency_vs_energy(chain, sims, band)
        effs.append(eff[0])
        errs.append(err[0])
    chain.threshold = saved                       # leave the chain at its operating point
    return np.array(out_rates), np.array(effs), np.array(errs)


def efficiency_model(energy_gev, a, b, c, d):
    """Empirical model of the trigger efficiency: log10(eff) = a - b * (1 / (log10(E) + c))^d."""
    return 10 ** (a - b * (1.0 / (np.log10(energy_gev) + c)) ** d)


def fit_efficiency(x, eff, err, p0=(-0.3, 1.0, 1.0, 2.0)):
    """Fit of ``efficiency_model`` to an efficiency curve (bins with eff > 0). Returns the parameters."""
    from scipy.optimize import curve_fit
    ok = (eff > 0) & (err > 0)
    params, _ = curve_fit(efficiency_model, x[ok], eff[ok], sigma=err[ok], p0=p0, maxfev=20000)
    return params