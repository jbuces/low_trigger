"""
Trigger chains: L1 (OR or TDSCAN) with or without a CNN, evaluated on gamma and NSB frames.
"""
import numpy as np

from . import rates
from .io import read_chain_table

# Samples where the first trigger of a shower is searched (15 to 25), as in the DLTriggerReader
SHOWER_WINDOW = (15, 26)


class SimulatedShowers:
    """
    Simulated gamma showers, used as denominator of the efficiencies.

    Parameters
    ----------
    energies : str or array
        Energy (GeV) of each simulated CORSIKA shower, or the path of a ``.npy`` file with them.
    reuse : int
        Number of times each shower is reused (different impact points).
    n_tel : int
        Number of telescopes per event.
    radius_m : float
        Radius of the simulated area (m).
    """

    def __init__(self, energies, reuse=10, n_tel=4, radius_m=800.0, runs=None):
        file_runs = None
        if isinstance(energies, str):
            data = np.load(energies)
            if isinstance(data, np.lib.npyio.NpzFile):
                energies, file_runs = data["energy"], data["run"]
            else:
                energies = data
        energies = np.asarray(energies, dtype=float)

        # Keep only the showers of the selected runs (needs the .npz with the run of each shower)
        if runs is not None:
            if file_runs is None:
                raise ValueError("Selecting runs needs the .npz file with the run of each shower.")
            runs = np.asarray(list(runs))
            absent = np.setdiff1d(runs, file_runs)
            if absent.size:
                raise ValueError(f"Runs {absent.tolist()} are not in the simulated showers file.")
            energies = energies[np.isin(file_runs, runs)]

        # Runs included (None if unknown, i.e. with the old .npy)
        if file_runs is None:
            self.runs = None
        elif runs is not None:
            self.runs = np.unique(runs)
        else:
            self.runs = np.unique(file_runs)

        self.energies = energies
        self.reuse = reuse
        self.n_tel = n_tel
        self.radius_m = radius_m

    @property
    def area_m2(self):
        return np.pi * self.radius_m**2

    @property
    def trials_per_shower(self):
        # Telescope frames per simulated shower (mono)
        return self.reuse * self.n_tel

    def counts(self, bins):
        """Number of simulated showers per energy bin (GeV)."""
        return np.histogram(self.energies, bins=bins)[0]


class TriggerChain:
    """
    A trigger chain evaluated on gamma and NSB frames.

    - Without CNN (L1 + OR, L1 + TDSCAN): a frame triggers if its trigger file has a trigger.
    - With CNN: the CNN also has to accept the frame, ``cnn_prediction <= threshold``
      (the prediction is the NSB score). The threshold is set with ``calibrate``.

    For gammas the L1 trigger must be in the shower window (samples 15-25). For NSB any
    trigger in the frame counts. Use the constructors ``TriggerChain.l1`` and ``TriggerChain.cnn``.
    """

    def __init__(self, name, gamma, nsb, has_cnn, shower_window=SHOWER_WINDOW, nsb_edge=0):
        self.name = name
        self.gamma = gamma
        self.nsb = nsb
        self.nsb_edge = nsb_edge
        self.has_cnn = has_cnn
        self.shower_window = shower_window
        self.threshold = None
        self._l1_rates = {}

    @classmethod
    def l1(cls, name, gamma_trigger_files, nsb_trigger_files, extra_columns=(), **kwargs):
        """Chain without CNN (L1 + OR, L1 + TDSCAN, ...)."""
        return cls(
            name,
            read_chain_table(gamma_trigger_files, extra_columns=extra_columns),
            read_chain_table(nsb_trigger_files),
            has_cnn=False,
            **kwargs,
        )

    @classmethod
    def cnn(cls, name, gamma_trigger_files, nsb_trigger_files,
            gamma_prediction_files, nsb_prediction_files, extra_columns=(), **kwargs):
        """Chain with CNN: the L1 trigger files and the CTLearn predictions made with them."""
        return cls(
            name,
            read_chain_table(gamma_trigger_files, gamma_prediction_files, extra_columns=extra_columns),
            read_chain_table(nsb_trigger_files, nsb_prediction_files),
            has_cnn=True,
            **kwargs,
        )

    # ---------- NSB rates ----------
    def l1_rate(self, definition="trigger"):
        """
        L1 rate on NSB (Hz). ``definition``: ``trigger`` (onsets per second, default),
        ``window`` (non overlapping 10-sample windows), ``poisson`` or ``frame`` (P / T).
        """
        if definition not in self._l1_rates:
            tps = self.nsb["trigger_per_sample"]
            if definition == "trigger":
                rate = rates.trigger_rate(tps, edge=self.nsb_edge)
            elif definition == "window":
                rate = rates.window_stats(tps, edge=self.nsb_edge)["window_rate"]
            elif definition == "poisson":
                rate = rates.poisson_rate(tps, edge=self.nsb_edge)
            elif definition == "frame":
                rate = rates.frame_rate(tps, edge=self.nsb_edge)
            else:
                raise ValueError(f"Unknown rate definition '{definition}'.")
            self._l1_rates[definition] = rate
        return self._l1_rates[definition]

    def _cnn_accepts(self, table):
        if self.threshold is None:
            raise RuntimeError(f"'{self.name}': call calibrate() first to set the CNN threshold.")
        prediction = np.asarray(table["cnn_prediction"], dtype=float)
        return np.isfinite(prediction) & (prediction <= self.threshold)

    def nsb_acceptance(self):
        """Fraction of the NSB windows evaluated by the CNN that it accepts."""
        evaluated = np.isfinite(np.asarray(self.nsb["cnn_prediction"], dtype=float))
        return self._cnn_accepts(self.nsb)[evaluated].mean()

    def output_rate(self, definition="trigger"):
        """NSB rate at the output of the chain (Hz)."""
        rate = self.l1_rate(definition)
        return rate * self.nsb_acceptance() if self.has_cnn else rate

    def calibrate(self, target_rate_kHz, definition="trigger"):
        """
        Set the CNN threshold that gives ``target_rate_kHz`` of NSB at the output.

        The CNN has to accept the fraction f = target / L1 rate of the NSB windows it
        evaluates, so the threshold is the f-quantile of their predictions.
        """
        if not self.has_cnn:
            raise ValueError(f"'{self.name}' has no CNN to calibrate.")
        l1_rate = self.l1_rate(definition)
        fraction = target_rate_kHz * 1e3 / l1_rate
        if not 0 < fraction <= 1:
            raise ValueError(f"{target_rate_kHz} kHz is not reachable with an L1 rate of {l1_rate / 1e3:.1f} kHz.")
        prediction = np.asarray(self.nsb["cnn_prediction"], dtype=float)
        self.threshold = float(np.quantile(prediction[np.isfinite(prediction)], fraction))
        print(
            f"{self.name}: L1 {l1_rate / 1e3:.1f} kHz -> CNN accepts {100 * fraction:.3f} % "
            f"-> threshold {self.threshold:.6f} -> output {self.output_rate(definition) / 1e3:.1f} kHz"
        )
        return self.threshold

    # ---------- Triggered frames ----------
    def gamma_triggered(self):
        """True for the gamma frames triggered by the chain."""
        tps = np.asarray(self.gamma["trigger_per_sample"])
        if self.shower_window is None:
            triggered = np.ones(len(tps), dtype=bool)
        else:
            start, stop = self.shower_window
            triggered = tps[:, start:stop].any(axis=1)
        if self.has_cnn:
            triggered &= self._cnn_accepts(self.gamma)
        return triggered

    def nsb_triggered(self):
        """True for the NSB frames triggered by the chain (any trigger in the frame)."""
        tps = np.asarray(self.nsb["trigger_per_sample"])
        e = self.nsb_edge
        triggered = tps[:, e:tps.shape[1] - e].any(axis=1)
        if self.has_cnn:
            triggered &= self._cnn_accepts(self.nsb)
        return triggered

    def gamma_table(self):
        """Gamma table with a ``triggered`` column."""
        out = self.gamma.copy(copy_data=False)
        out["triggered"] = self.gamma_triggered()
        return out

    def __repr__(self):
        cnn = f", CNN threshold {self.threshold}" if self.has_cnn else ""
        return f"TriggerChain('{self.name}', {len(self.gamma)} gamma / {len(self.nsb)} NSB frames{cnn})"
