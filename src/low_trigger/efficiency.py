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


def crab_magic(energy_gev):
    """Crab Nebula spectrum (MAGIC, Aleksić et al. 2015, log-parabola): dN/dE in TeV^-1 cm^-2 s^-1."""
    e = np.asarray(energy_gev, dtype=float) / 1e3
    return 3.23e-11 * e ** (-2.47 - 0.24 * np.log10(e))


def power_law(index, norm=3.23e-11, e0_gev=1000.0):
    """Power-law spectrum dN/dE = norm * (E / E0)^-index, in TeV^-1 cm^-2 s^-1."""
    return lambda energy_gev: norm * (np.asarray(energy_gev, dtype=float) / e0_gev) ** (-index)


def differential_rate(chain, sims, bins, spectrum=crab_magic, fit=False):
    """
    Gamma-ray trigger rate per decade of energy, dR/dlog10(E) = A_eff(E) * dN/dE * E * ln(10), in Hz.
    Its maximum is the energy threshold of the chain. With ``fit``, the effective area comes from the
    empirical fit of the efficiency (smooth curve on a fine energy grid, no error).
    Returns the energies (GeV), the rate and its error.
    """
    x, eff, err = efficiency_vs_energy(chain, sims, bins)
    if fit:
        params = fit_efficiency(x, eff, err)
        x = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 400)
        eff, err = efficiency_model(x, *params), np.zeros(len(x))
    area_cm2 = sims.area_m2 * 1e4
    factor = area_cm2 * spectrum(x) * (x / 1e3) * np.log(10)        # dN/dE in TeV^-1, E in TeV
    return x, eff * factor, err * factor


def energy_threshold(x, rate):
    """Energy (GeV) of the maximum of the differential rate."""
    return float(x[np.argmax(rate)])

def proton_spectrum(energy_gev):
    """Cosmic-ray proton spectrum (BESS, as used in CTA): dN/dE in TeV^-1 cm^-2 s^-1 sr^-1."""
    e = np.asarray(energy_gev, dtype=float) / 1e3
    return 9.6e-6 * e ** -2.70


def cone_solid_angle(radius_deg):
    """Solid angle (sr) of a cone of the given half-opening angle."""
    return 2 * np.pi * (1 - np.cos(np.radians(radius_deg)))


def cosmic_ray_rate(chain, sims, bins, spectrum=proton_spectrum, solid_angle_sr=cone_solid_angle(2.25),
                    light_factor=1.0):
    """
    Trigger rate (Hz) of a diffuse cosmic-ray flux: R = sum over bins of eff * A_sim * Omega * J(E) * dE.

    - With simulated protons (diffuse within a view cone): light_factor=1 and Omega = solid angle of the
      simulated view cone.
    - With gamma rays as a proxy: light_factor ~ 3 (a proton of energy E gives about as much Cherenkov light
      as a gamma ray of energy E / 3), and Omega = solid angle of the field of view (default: 2.25 deg radius).
      This is a rough estimate (on-axis gamma rays, CNN not trained on hadrons).
    For mono chains the rate is per telescope.
    """
    _, eff, _ = efficiency_vs_energy(chain, sims, bins)
    edges = np.asarray(bins, dtype=float) * light_factor          # proton energies of the bins (GeV)
    centers = np.sqrt(edges[:-1] * edges[1:])
    de_tev = np.diff(edges) / 1e3
    return float(np.sum(eff * sims.area_m2 * 1e4 * solid_angle_sr * spectrum(centers) * de_tev))


def rate_vs_threshold(chain, sims, bins, thresholds=None, n_points=40, **cosmic_kwargs):
    """
    NSB output rate and cosmic-ray rate (Hz) for a range of thresholds of the chain: the CNN threshold
    for chains with CNN (chosen from the NSB score distribution), or ``thresholds`` for a ScanChain.
    Returns the thresholds, the NSB rates and the cosmic-ray rates.
    """
    if getattr(chain, "has_cnn", False) and thresholds is None:
        p = np.asarray(chain.nsb["cnn_prediction"], dtype=float)
        thresholds = np.quantile(p[np.isfinite(p)], np.linspace(0.002, 1.0, n_points))
    elif thresholds is None:
        raise ValueError("Give the OR thresholds to scan.")
    saved = chain.threshold
    nsb, cosmic = [], []
    for thr in thresholds:
        chain.threshold = thr
        nsb.append(chain.output_rate())
        cosmic.append(cosmic_ray_rate(chain, sims, bins, **cosmic_kwargs))
    chain.threshold = saved
    return np.asarray(thresholds, dtype=float), np.array(nsb), np.array(cosmic)