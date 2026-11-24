"""
Figures of the trigger chains, in paper style.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .efficiency import (energy_bins, efficiency_vs_energy, efficiency_vs_pe, efficiency_vs_rate,
                         efficiency_model, fit_efficiency, differential_rate, energy_threshold,
                         crab_magic, rate_vs_threshold)
PAPER_STYLE = {
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 18, "axes.titlesize": 18,
    "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 16, "xtick.major.size": 8, "xtick.minor.size": 4,
    "ytick.major.size": 8, "ytick.minor.size": 4, "xtick.major.width": 1.5,
    "ytick.major.width": 1.5, "xtick.direction": "in", "ytick.direction": "in",
}


def _figure(with_ratio):
    if not with_ratio:
        fig, ax = plt.subplots(figsize=(8, 7))
        return fig, ax, None
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    ax.tick_params(labelbottom=False, which="both")
    return fig, ax, axr


def _draw(ax, axr, x, curves, names, ratio_to, errors, colors, styles):
    for i, (y, err) in enumerate(curves):
        style = {"color": colors[i] if colors else f"C{i}", "linestyle": styles[i] if styles else "solid"}
        if errors:
            ax.errorbar(x, y, yerr=err, capsize=2, label=names[i], **style)
        else:
            ax.plot(x, y, label=names[i], **style)
        if axr is not None:
            ref = curves[ratio_to][0]
            axr.plot(x, np.divide(y, ref, out=np.full_like(y, np.nan), where=ref > 0), **style)


def _finish(fig, ax, axr, xlabel, ylabel, ylim, save):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    (axr if axr is not None else ax).set_xlabel(xlabel)
    if axr is not None:
        axr.set_ylabel("Ratio")
        axr.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    if save:
        for fmt in ("jpg", "pdf"):
            fig.savefig(f"{save}.{fmt}", dpi=300, bbox_inches="tight")
    plt.show()


def plot_efficiency(chains, sims, bins=None, effective_area=False, ratio_to=None, level="shower",
                    errors=True, ymin=None, colors=None, styles=None, save=None, fit=False):
    """
    Trigger efficiency (or effective area) vs true energy of several chains.
    ``ratio_to``: index of the reference chain for the ratio panel (None: no ratio).
    """
    bins = energy_bins() if bins is None else bins
    scale = sims.area_m2 if effective_area else 1.0
    curves = []
    for chain in chains:
        x, eff, err = efficiency_vs_energy(chain, sims, bins, level=level)
        curves.append((eff * scale, err * scale))

    with plt.rc_context(PAPER_STYLE):
        fig, ax, axr = _figure(ratio_to is not None)
        names = [c.name for c in chains]
        if fit:
            # Data as markers, fitted curves as lines; the ratio panel uses the fits
            x_fit = np.logspace(np.log10(x[0]), np.log10(x[-1]), 300)
            fits = [(efficiency_model(x_fit, *fit_efficiency(x, y / scale, e / scale)) * scale, None)
                    for y, e in curves]
            for i, (y, e) in enumerate(curves):
                ax.errorbar(x, y, yerr=e, fmt="o", markersize=3, alpha=0.5,
                            color=colors[i] if colors else f"C{i}")
            _draw(ax, axr, x_fit, fits, names, ratio_to, False, colors, styles)
        else:
            _draw(ax, axr, x, curves, names, ratio_to, errors, colors, styles)
        ylabel = r"$A_\mathrm{eff}$ [m$^2$]" if effective_area else "Trigger efficiency"
        ylim = (ymin, None) if effective_area else (ymin, 1.05)
        _finish(fig, ax, axr, "True energy [GeV]", ylabel, ylim, save)
    return x, curves


def plot_efficiency_vs_pe(chains, bins=None, ratio_to=None, errors=True, ymin=None,
                          colors=None, styles=None, save=None):
    """Trigger efficiency vs true Cherenkov p.e. in the camera of several chains."""
    bins = np.logspace(0, 4, 41) if bins is None else bins
    curves = []
    for chain in chains:
        x, eff, err = efficiency_vs_pe(chain, bins)
        curves.append((eff, err))

    with plt.rc_context(PAPER_STYLE):
        fig, ax, axr = _figure(ratio_to is not None)
        _draw(ax, axr, x, curves, [c.name for c in chains], ratio_to, errors, colors, styles)
        _finish(fig, ax, axr, "True Cherenkov p.e.", "Trigger efficiency", (ymin, 1.05), save)
    return x, curves

def plot_efficiency_vs_rate(chains, sims, energy_range=(10.0, 30.0), or_thresholds=None,
                            operating_rate_hz=None, colors=None, save=None):
    """
    Trigger efficiency in an energy band vs NSB output rate. CNN chains give a curve (moving the CNN
    threshold), ScanChains a curve over ``or_thresholds[name]``, and other chains a single point.
    """
    or_thresholds = or_thresholds or {}
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))
        for i, chain in enumerate(chains):
            r, eff, err = efficiency_vs_rate(chain, sims, energy_range, thresholds=or_thresholds.get(chain.name))
            style = {"color": colors[i] if colors else f"C{i}", "label": chain.name}
            if len(r) > 1:
                ax.plot(r / 1e6, eff, **style)
            else:
                ax.errorbar(r / 1e6, eff, yerr=err, fmt="o", markersize=8, **style)
        if operating_rate_hz:
            ax.axvline(operating_rate_hz / 1e6, color="k", linestyle=":", label="Operating point")
        ax.set_xscale("log")
        ax.set_xlabel("NSB output rate [MHz]")
        ax.set_ylabel(f"Trigger efficiency ({energy_range[0]:g}–{energy_range[1]:g} GeV)")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=11)
        fig.tight_layout()
        if save:
            for fmt in ("jpg", "pdf"):
                fig.savefig(f"{save}.{fmt}", dpi=300, bbox_inches="tight")
        plt.show()


def plot_score_distributions(chains, save=None):
    """Distributions of the CNN score (NSB score) on gamma rays and NSB, with the threshold of each chain."""
    chains = [c for c in chains if c.has_cnn]
    fig, axes = plt.subplots(1, len(chains), figsize=(7 * len(chains), 5), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], chains):
        for table, label in ((c.gamma, "gamma rays"), (c.nsb, "NSB")):
            p = np.asarray(table["cnn_prediction"], dtype=float)
            p = p[np.isfinite(p)]
            ax.hist(p, bins=100, range=(0, 1), histtype="step", density=True, log=True, label=f"{label} ({len(p)})")
        if c.threshold is not None:
            ax.axvline(c.threshold, color="k", linestyle="--", label=f"threshold {c.threshold:.3f}")
        ax.set_title(c.name, fontsize=11)
        ax.set_xlabel("CNN prediction (NSB score)")
        ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f"{save}.pdf", bbox_inches="tight")
    plt.show()


def plot_data_distribution(chains, sims, bins=None, unit="event", save=None):
    """
    Simulated and triggered counts per energy bin. ``unit``:
    - 'shower': CORSIKA showers (triggered: at least one triggered frame among all reuses and telescopes);
    - 'event': shower reuses, i.e. array events (triggered: at least one triggered telescope);
    - 'frame': telescope frames (simulated: showers x reuses x telescopes).
    """
    bins = energy_bins() if bins is None else bins
    centers = np.sqrt(bins[:-1] * bins[1:])
    factor = {"shower": 1, "event": sims.reuse, "frame": sims.reuse * sims.n_tel}[unit]
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(centers, sims.counts(bins) * factor, where="mid", lw=2, linestyle="--", color="k", label="Simulated")
        for i, c in enumerate(chains):
            t = c.gamma
            trig = c.gamma_triggered()
            energy = np.asarray(t["true_energy"])[trig] * 1e3
            if unit != "frame":
                key = np.asarray(t["event_id"])[trig] // (100 if unit == "shower" else 1)
                _, first = np.unique(np.stack([np.asarray(t["obs_id"])[trig], key], axis=1), axis=0, return_index=True)
                energy = energy[first]
            ax.step(centers, np.histogram(energy, bins=bins)[0], where="mid", lw=2, color=f"C{i}", label=c.name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("True energy [GeV]")
        ax.set_ylabel({"shower": "Showers", "event": "Events", "frame": "Telescope frames"}[unit])
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=12)
        fig.tight_layout()
        if save:
            fig.savefig(f"{save}.pdf", bbox_inches="tight")
        plt.show()


def plot_feb_coincidences(counts, t1, t2, feb, geometry, save=None):
    """
    FEB ``feb`` of telescope t1, and the distribution (in % and in counts) of the FEBs of t2 where the
    same showers fall, from the counts of make_feb_coincidences.py. ``geometry``: ctapipe CameraGeometry.
    """
    from ctapipe.visualization import CameraDisplay
    c = counts[(t1, t2)][feb]
    image = np.zeros(163 * 49)
    image[feb * 49:(feb + 1) * 49] = 1                        # the 49 pixels of each FEB are consecutive
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, values, title in zip(axes, (image, np.repeat(100 * c / max(c.sum(), 1), 49), np.repeat(c, 49)),
                                 (f"Tel {t1}, FEB {feb}", f"Tel {t2}, FEBs (% of showers)", f"Tel {t2}, FEBs (counts)")):
        disp = CameraDisplay(geometry, ax=ax)
        disp.image = values
        disp.add_colorbar()
        ax.set_title(title)
    fig.tight_layout()
    if save:
        fig.savefig(f"{save}.pdf", bbox_inches="tight")
    plt.show()


def plot_differential_rate(chains, sims, bins=None, spectrum=None, fit=False, normalize=False,
                           colors=None, styles=None, save=None):
    """
    Differential gamma-ray trigger rate (per decade of energy) for a reference spectrum, and the energy
    threshold (maximum) of each chain. ``sims`` can be a list with one SimulatedShowers per chain
    (e.g. n_tel=4 for mono chains, n_tel=1 for stereo chains). With ``normalize``, each curve is scaled
    to its maximum, to compare the shapes.
    """
    bins = energy_bins(e_max=1e4) if bins is None else bins
    spectrum = crab_magic if spectrum is None else spectrum
    sims_list = sims if isinstance(sims, (list, tuple)) else [sims] * len(chains)
    thresholds = {}
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, (chain, s) in enumerate(zip(chains, sims_list)):
            x, rate, err = differential_rate(chain, s, bins, spectrum, fit=fit)
            e_thr = energy_threshold(x, rate)
            thresholds[chain.name] = e_thr
            scale = 1.0 / rate.max() if normalize else 1.0
            color = colors[i] if colors else f"C{i}"
            style = styles[i] if styles else "solid"
            label = f"{chain.name}: $E_\\mathrm{{th}}$ = {e_thr:.0f} GeV"
            if fit:
                ax.plot(x, rate * scale, color=color, linestyle=style, label=label)
            else:
                ax.errorbar(x, rate * scale, yerr=err * scale, color=color, linestyle=style, capsize=2, label=label)
            ax.axvline(e_thr, color=color, linestyle=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("True energy [GeV]")
        ax.set_ylabel("Normalised rate" if normalize else r"$dR/d\log_{10}E$ [Hz]")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=12)
        fig.tight_layout()
        if save:
            for fmt in ("jpg", "pdf"):
                fig.savefig(f"{save}.{fmt}", dpi=300, bbox_inches="tight")
        plt.show()
    return thresholds

def _plot_bias(thr, nsb, cosmic, operating, title, xlabel, save):
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(thr, nsb, label="NSB")
        ax.plot(thr, cosmic, label="Cosmic rays")
        ax.plot(thr, nsb + cosmic, color="k", linestyle="--", label="Total")
        if operating is not None:
            ax.axvline(operating, color="k", linestyle=":", label="Operating threshold")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Trigger rate [Hz]")
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        if save:
            for fmt in ("jpg", "pdf"):
                fig.savefig(f"{save}.{fmt}", dpi=300, bbox_inches="tight")
        plt.show()


def plot_rate_vs_threshold(chain, sims, bins=None, thresholds=None, xlabel=None, save=None, **cosmic_kwargs):
    """Mono trigger rate vs threshold ('bias curve'): NSB, cosmic rays and their sum."""
    bins = energy_bins(e_max=1e4) if bins is None else bins
    thr, nsb, cosmic = rate_vs_threshold(chain, sims, bins, thresholds, **cosmic_kwargs)
    _plot_bias(thr, nsb, cosmic, chain.threshold, chain.name,
               xlabel or ("CNN threshold (NSB score)" if chain.has_cnn else "OR threshold"), save)
    return thr, nsb, cosmic


def plot_stereo_rate_vs_threshold(mono, sims_evt, thresholds, bins=None, min_tels=2, lut=None, title=None,
                                  xlabel="OR threshold", save=None, **cosmic_kwargs):
    """Array trigger rate vs threshold of the mono chain, after the stereo (multiplicity or topological) trigger."""
    from .stereo import stereo_rate_vs_threshold
    bins = energy_bins(e_max=1e4) if bins is None else bins
    thr, nsb, cosmic = stereo_rate_vs_threshold(mono, sims_evt, bins, thresholds, min_tels, lut, **cosmic_kwargs)
    kind = "topological" if lut is not None else "multiplicity"
    _plot_bias(thr, nsb, cosmic, mono.threshold, title or f"{mono.name} + stereo ({kind})", xlabel, save)
    return thr, nsb, cosmic