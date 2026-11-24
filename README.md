# low_trigger

Trigger studies for the **LST Advanced Camera (AdvCam)**: Level 1 (L1) digital sums, Level 2 (L2) algorithms
(OR, TDSCAN, OR + CNN), NSB trigger rates, gamma-ray trigger efficiencies and effective areas, threshold
scans and the stereo trigger.

The workflow has three stages:

1. **Trigger files** (`scripts/get_l1tdscan.py`): apply the L1 (and OR / TDSCAN) to the R0 waveforms of the
   simulations and store, for every telescope frame, the trigger per sample and other information.
2. **CNN predictions** (CTLearn + DL1DataHandler `DLTriggerReader`): evaluate the CNN on the windows selected
   by the OR, using the trigger files of step 1.
3. **Analysis** (`low_trigger` package): build trigger chains from the trigger files and the predictions,
   calibrate them to a common NSB output rate, and compute rates, efficiencies and figures.

---

## Repository structure

```
low_trigger/
├── src/low_trigger/       Python package (analysis)
│   ├── io.py              reading of trigger files and CTLearn predictions
│   ├── rates.py           NSB trigger rates (several definitions)
│   ├── chain.py           TriggerChain (L1 / L1 + CNN) and SimulatedShowers (denominator)
│   ├── efficiency.py      efficiency vs energy / p.e. / NSB rate, empirical fit
│   ├── plotting.py        figures (paper style)
│   ├── scan.py            threshold scans directly from the R0 waveforms
│   ├── stereo.py          stereo trigger (multiplicity and topological)
│   └── legacy/            old analysis modules (kept for reference)
├── scripts/
│   ├── get_l1tdscan.py           trigger files from the R0 waveforms
│   ├── get_sim_energies.py       energies of all simulated CORSIKA showers (denominator)
│   └── make_feb_coincidences.py  FEB coincidence counts for the topological stereo trigger
├── condor/                example HTCondor submit files
├── resources/
│   └── triggergeometry_AdvCam_v1.h5   AdvCam trigger geometry (flowers, FEBs, sectors, neighbours)
└── pyproject.toml
```

## Installation

```bash
git clone <repo-url> low_trigger
cd low_trigger
pip install -e . --no-deps     # in an environment with numpy, scipy, pandas, matplotlib, astropy, tables, ctapipe
```

`--no-deps` avoids changing the versions of an existing environment (e.g. the CTLearn one).
`scripts/get_sim_energies.py` only needs `numpy` and `eventio` and can run in a minimal virtual environment.

---

## 1. Trigger files: `scripts/get_l1tdscan.py`

One output file per input R0 file. Example (amplitude-aware L1 with OR on the FEB amplitude):

```bash
python scripts/get_l1tdscan.py \
    -i /path/gammas/corsika_run1.r0.h5 \
    -o /path/trigger/run1/quantised_l1/step1_2160_47/corsika_run1_2160cut.h5 \
    -nf resources/triggergeometry_AdvCam_v1.h5 \
    -trig quantised_l1 -l1 2160 -pts 47 -epsl1 1 -epst 1 -qstep 1 -add False
```

| Option | Meaning |
|---|---|
| `-trig` | `l1` (Boolean OR), `quantised_l1` (amplitude aware), `tdscan`, `tdscan_perflower`, `rolling_l1`, `quantised_rolled_digital_sum`, `quantised_per_feb` |
| `-l1` | L1 threshold of the digital sum (ADC counts); the comparison is `>=` |
| `-epsl1` | digital-sum region: 1 flower (7 pixels), 2 superflower (49, overlapping), 3 FEB |
| `-pts` | `quantised_l1`: OR threshold on the FEB amplitude (`min_pts`); `tdscan`: minimum points |
| `-epst` | TDSCAN half window in samples |
| `-qstep` | quantisation step of the amplitudes (0 for the Boolean modes) |
| `-add` | `True` to store the packed trigger mask (needed only for `mask_from_file` in the reader) |
| `-fw` | window (first and last sample) of `triggered_febs`, default `15 25` (shower window) |
| `-nf` | trigger geometry file |

Main columns of the output table (`/table`):

| Column | Content |
|---|---|
| `obs_id`, `event_id`, `tel_id`, `table_index` | frame identifiers (`table_index`: row in the R0 table) |
| `true_energy` (TeV), `true_image_sum` (p.e.) | simulated shower |
| `trigger_per_sample` (40) | OR output per sample: 1 if any flower / FEB is on |
| `trigger_per_sector` (3, 40) | the same per camera sector |
| `low_trigger` | any trigger in the frame |
| `triggered_febs` (163) | FEBs with a trigger inside the `-fw` window |
| `max_pix`, `peak_sample` | brightest pixel of the true image and its peak sample |

The table attribute `input_file` stores the name of the input file; the `DLTriggerReader` uses it to pair
trigger files and input files.

## 2. Simulated showers: `scripts/get_sim_energies.py`

The denominator of the efficiencies is the number of **simulated** showers, not the stored ones (sim_telarray
only stores frames above 20 p.e.). This script reads the `EventHeader` of the CORSIKA files (one per shower,
not per reuse) and writes an `.npz` with `energy` (GeV) and `run`:

```bash
python scripts/get_sim_energies.py -i "/path/corsika_run{run}.corsika.gz" -r 1 50 -o sim_energies.npz
```

## 3. CNN predictions (CTLearn)

The CNN is trained and evaluated with CTLearn and the `DLTriggerReader` of DL1DataHandler, which takes the
trigger files of step 1 (`--DLTriggerReader.input_trigger_files`) and builds a 10-sample window starting one
sample before the first OR trigger (gamma rays: searched in samples 15–25; NSB: anywhere in the frame).
Each model is predicted on the trigger files of its own OR threshold. The prediction files are read by
`low_trigger.io.read_predictions`.

---

## 4. Analysis

### Trigger chains

```python
from low_trigger.io import file_list
from low_trigger.chain import TriggerChain, SimulatedShowers

GAMMA_RUNS, NSB_RUNS = range(1, 51), range(21, 26)
gammas = lambda p: file_list(p, GAMMA_RUNS)     # raises an error listing the missing files
nsb = lambda p: file_list(p, NSB_RUNS)

# L1 + OR (no CNN)
or_flower = TriggerChain.l1(
    "L1 bool. + OR",
    gammas("/path/gammas/trigger/run{run}/l1/2221/corsika_run{run}_2221.h5"),
    nsb("/path/nsb/trigger/run{run}/l1/2221/corsika_run{run}_2221.h5"),
)

# L1 + OR + CNN
cnn = TriggerChain.cnn(
    "L1 amp. aware + 2D CNN",
    gammas("/path/gammas/trigger/run{run}/quantised_l1/step1_2160_47/corsika_run{run}_2160cut.h5"),
    nsb("/path/nsb/trigger/run{run}/quantised_l1/step1_2160_47/corsika_run{run}_2160cut.h5"),
    gammas("/path/predictions/<model>/gammas/run{run}/predict.h5"),
    nsb("/path/predictions/<model>/nsb/run{run}/predict.h5"),
)
cnn.calibrate(or_flower.output_rate() / 1e3)    # CNN threshold for the same NSB output rate (kHz)

# Denominator: the simulated showers of the same gamma runs
sims = SimulatedShowers("sim_energies.npz", runs=GAMMA_RUNS)
```

- A **gamma** frame triggers if the OR has a trigger in the shower window (samples 15–25) and, with a CNN,
  if the CNN accepts it (`cnn_prediction <= threshold`; the prediction is the NSB score).
- `calibrate(target_kHz)` sets the CNN threshold as the quantile of the NSB predictions that gives the
  target output rate.
- For TDSCAN, use `nsb_edge=1`: the first and last samples use a truncated window and are dropped from the
  NSB rates.
- Extra columns of the trigger files can be read with `extra_columns=("triggered_febs",)`.

### NSB rates (`rates.py`)

| Function | Definition |
|---|---|
| `trigger_rate` | **onsets per second** (0 → 1 transitions in the stream of all frames); a trigger lasting several samples counts once. This is the CNN input rate when every OR trigger is evaluated. **Default.** |
| `window_stats` | non-overlapping windows opened by the triggers, fraction with a second trigger, busy fraction |
| `poisson_rate` | −ln(1 − P) / T, with P the fraction of frames with a trigger |
| `frame_rate` | P / T (saturates at 25 MHz for 40-ns frames; only for comparison with old numbers) |

`chain.l1_rate(definition)` gives the OR rate and `chain.output_rate()` the rate after the CNN.

### Efficiencies and figures

```python
from low_trigger.efficiency import energy_bins, efficiency_vs_energy, efficiency_vs_rate
from low_trigger.plotting import (plot_efficiency, plot_efficiency_vs_pe, plot_efficiency_vs_rate,
                                  plot_score_distributions, plot_data_distribution)

bins = energy_bins(e_max=1e4)
chains = [or_flower, cnn]
plot_efficiency(chains, sims, bins=bins, ratio_to=0)                        # efficiency vs energy
plot_efficiency(chains, sims, bins=bins, effective_area=True, ratio_to=0)   # effective area
plot_efficiency(chains, sims, bins=bins, ratio_to=0, fit=True)              # with the empirical fit
plot_efficiency_vs_pe(chains, ratio_to=0)                                   # efficiency vs true p.e.
plot_efficiency_vs_rate(chains, sims, energy_range=(10, 30))                # efficiency vs NSB output rate
plot_score_distributions([cnn])                                             # CNN score, gammas and NSB
plot_data_distribution(chains, sims, bins=bins, unit="frame")               # simulated vs triggered counts
```

- The efficiency is the fraction of simulated telescope frames (showers × reuses × telescopes) that trigger.
  Its error treats each **shower** as an independent unit (`level="shower"`, default), since the frames of the
  same shower (4 telescopes × 10 reuses) are correlated. Showers are identified by `event_id // 100`.
- `SimulatedShowers(..., runs=...)` must contain exactly the gamma runs of the chains; otherwise an error is
  raised.
- `plot_data_distribution(..., unit=...)` counts `"shower"` (CORSIKA showers, default), `"event"` (shower
  reuses, i.e. array events) or `"frame"` (telescope frames, the unit of the mono efficiency).
- `plot_efficiency(..., fit=True)` draws the data as markers and the empirical fit
  log10(ε) = A − B (1 / (log10 E + C))^D as lines; the ratio panel uses the fits.
- `efficiency_vs_rate` / `plot_efficiency_vs_rate`: efficiency in an energy band vs NSB output rate. CNN chains
  give a curve (moving the CNN threshold), `ScanChain`s a curve over the OR thresholds given in
  `or_thresholds={chain name: thresholds}`, and chains from trigger files without CNN a single point.

### Energy threshold

The energy threshold is the maximum of the differential gamma-ray rate for a reference spectrum,
dR/dlog10(E) = A_eff(E) · dN/dE · E · ln 10 (Crab Nebula, MAGIC log-parabola, by default):

```python
from low_trigger.efficiency import differential_rate, energy_threshold, crab_magic, power_law
from low_trigger.plotting import plot_differential_rate

# One SimulatedShowers per chain: n_tel=4 for mono chains (A_eff per telescope), n_tel=1 for stereo chains
thresholds = plot_differential_rate([mono, stereo_mult, stereo_topo], [sims, sims_evt, sims_evt],
                                    fit=True, normalize=True)                  # {chain name: E_th (GeV)}
plot_differential_rate(chains, sims, spectrum=power_law(2.6), fit=True)
```

With `fit=True` the effective area comes from the empirical fit, so the maximum is well defined even with
limited statistics; with `fit=False` the binned values are used (check that both give the same peak).

### Trigger rate vs threshold (bias curves)

NSB rate (steep) and cosmic-ray rate (shallow) vs the threshold of the chain, mono or after the stereo trigger:

```python
from low_trigger.plotting import plot_rate_vs_threshold, plot_stereo_rate_vs_threshold

plot_rate_vs_threshold(or_scan, sims, thresholds=range(2190, 2260, 2), light_factor=3)      # mono, per telescope
plot_stereo_rate_vs_threshold(or_scan, sims_evt, range(2190, 2260, 2), lut=lut, light_factor=3)   # array

# CNN: scan the CNN threshold (the rates increase with it, since frames are accepted below it)
p = np.asarray(cnn.nsb["cnn_prediction"], dtype=float)
cnn_thresholds = np.quantile(p[np.isfinite(p)], np.linspace(0.002, 1.0, 30))
plot_stereo_rate_vs_threshold(cnn, sims_evt, cnn_thresholds, lut=lut, xlabel="CNN threshold (NSB score)",
                              light_factor=3)
```

The cosmic-ray rate is R = Σ ε(E) · A_sim · Ω · J(E) · ΔE with the proton spectrum J(E) (BESS, as used in CTA):

- **With gamma rays as a proxy** (`light_factor=3`, Ω = solid angle of the field of view, 2.25° radius by
  default): a proton of energy E is assumed to produce as much Cherenkov light as a gamma ray of energy E/3.
  Rough estimate only (on-axis gamma rays, CNN not trained on hadrons).
- **With simulated protons:** build the chain and `SimulatedShowers` from the proton files and use
  `light_factor=1` and `solid_angle_sr=cone_solid_angle(<view cone radius>)`.

For the stereo curves, the stereo chain is rebuilt for every threshold, so the topological condition needs the
triggered FEBs for each threshold: `ScanChain` provides them from the per-FEB maxima of the scan, and chains
from trigger files read the `triggered_febs` column (with `extra_columns=("triggered_febs",)`).

### Threshold scans from the R0 waveforms (`scan.py`)

A single pass over the waveforms stores, per frame, the maximum flower sum (Boolean OR) and the maximum FEB
amplitude (amplitude-aware OR), over the camera and for each FEB (in the shower window for gamma rays). Any
OR threshold can then be applied without generating trigger files, including the triggered FEBs needed by the
topological stereo trigger. Needs access to the R0 files.

```python
from low_trigger.scan import (load_flower_pixels, scan_nsb, scan_gamma, rates_vs_threshold,
                              threshold_for_rate, ScanChain)

fp = load_flower_pixels("resources/triggergeometry_AdvCam_v1.h5")
nsb_scan = scan_nsb(NSB_R0_FILES, fp, t_flower=2160)
gamma_scan = scan_gamma(GAMMA_R0_FILES, fp, t_flower=2160)

thr, rate = threshold_for_rate(nsb_scan["flower"], range(2180, 2260), target_hz=2e6)
or_scan = ScanChain("Boolean OR", gamma_scan, nsb_scan, "flower", thr)      # behaves like a chain
plot_efficiency_vs_rate([or_scan, cnn], sims, or_thresholds={"Boolean OR": range(2200, 2240)})
```

The threshold given to `ScanChain` is its operating threshold (used by the efficiency and plotting
functions); the functions that scan thresholds restore it at the end. Below the maximum of the
rate-vs-threshold curve the OR is almost always on and the onset rate is not meaningful;
`threshold_for_rate` only searches above that maximum. The scan reproduces exactly the trigger files
(same rate, frames and efficiency for the Boolean OR 2221).

### Stereo trigger (`stereo.py`)

**FEB coincidence table.** For each event of a diffuse gamma-ray simulation, the FEB of the brightest pixel of
each telescope image is taken; for every ordered pair of telescopes the script counts the pairs of FEBs:

```bash
python scripts/make_feb_coincidences.py -i "/path/gamma_diffuse_run{run}.r0.h5" -r 1 20 -o feb_coincidences.npz
```

The counts are generated once; the look-up table for any threshold is built from them:

```python
from low_trigger.stereo import (load_feb_counts, feb_lut, lut_summary, save_feb_lut,
                                mean_coincident_febs, StereoChain, stereo_nsb_rate)
from low_trigger.plotting import plot_feb_coincidences

counts = load_feb_counts("feb_coincidences.npz")
lut_summary(counts, thresholds=(0.0, 0.00065, 0.01, 0.05))   # mean coincident FEBs per pair and threshold
lut = feb_lut(counts, threshold=0.05)                        # FEB f2 allowed if > 5% of the showers of f1
save_feb_lut(lut, "feb_lut_5pct.npz")
plot_feb_coincidences(counts, 1, 2, feb=60, geometry=geom)   # visual check of one FEB
```

**Stereo chains.** Built on any mono chain (read with `extra_columns=("triggered_febs",)` for the
topological condition, or a `ScanChain`). With a look-up table, the topological factor f (mean coincident
FEBs / 163) is computed from the table itself. The efficiency is per event, so the denominator uses `n_tel=1`:

```python
mono = TriggerChain.l1("OR", gamma_files, nsb_files, extra_columns=("triggered_febs",))

stereo_mult = StereoChain(mono, min_tels=2, name="OR + stereo (multiplicity)")
stereo_topo = StereoChain(mono, min_tels=2, lut=lut, name="OR + stereo (topological)")

sims_evt = SimulatedShowers("sim_energies.npz", runs=GAMMA_RUNS, n_tel=1)
plot_efficiency([stereo_mult, stereo_topo], sims_evt, ratio_to=0)    # efficiency lost with the topological condition
print(f"NSB stereo rate: multiplicity {stereo_mult.output_rate():.0f} Hz, topological {stereo_topo.output_rate():.0f} Hz")
```

The NSB stereo rate is analytic, from the mono output rate R of each telescope:

  R_k = C(n, k) · k · R^k · τ^(k−1) · f^(k−1)

with n telescopes, k required, τ the coincidence half window (20 ns) and f the topological factor
(mean coincident FEBs / 163; f = 1 without the topological condition).

---

## Conventions and pitfalls

- **Threshold comparison:** the current scripts use `>=`. Trigger files made with old versions used `>`:
  an old file with threshold `X` corresponds to a new one with `X + 1` (e.g. old 2220 = new 2221).
- **Shower window:** samples 15–25 (`SHOWER_WINDOW = (15, 26)` in `chain.py`), where the simulation places the
  Cherenkov signal of the 40-sample readout.
- **Rates:** always quote the **trigger (onset) rate**; the frame rate P/T saturates and underestimates.
- **NSB files** contain only telescope 1 (single-telescope dummy CORSIKA file at the array origin).
- **Stored gamma frames:** only frames with ≥ 20 true p.e. are stored by sim_telarray; the efficiency
  denominator comes from `sim_energies.npz`, not from the stored frames.
- **Training vs test:** gamma runs 110–127 and NSB runs 1–7 for training; gamma runs 1–50 and NSB runs
  21–25 for test (independent runs, so all reuses of a shower are in the same sample).
- **`triggered_febs`:** trigger files made before the `-fw` option store the FEBs triggered in the whole
  40-sample frame; current ones only those in the shower window (samples 15–25).
- **Pixel numbering:** the 7 pixels of each flower and the 7 flowers (49 pixels) of each FEB are consecutive,
  so pixel p belongs to flower p // 7 and FEB p // 49.

## Reference values (NSB 268 MHz/pixel, NSB runs 21–25)

Trigger (onset) rate of the OR for the thresholds used in the trigger studies.

**Amplitude-aware OR** (flower threshold 2160, OR threshold `min_pts` on the FEB amplitude):

| `min_pts` | 40 | 41 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 54 | 55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rate (MHz) | 37.8 | 33.4 | 22.9 | 20.1 | 17.7 | 15.5 | 13.6 | 11.9 | 10.4 | 6.1 | 5.3 |

**Boolean OR** (threshold on the flower sum):

| threshold | 2200 | 2202 | 2204 | 2208 | 2209 | 2213 | 2215 | 2216 | 2220 | 2221 |
|---|---|---|---|---|---|---|---|---|---|---|
| rate (MHz) | 35.4 | 27.5 | 21.2 | 12.4 | 10.8 | 6.2 | 4.7 | 4.1 | 2.3 | 1.97 |

**Stereo trigger** (Boolean OR 2221, mono output 1.97 MHz per telescope, 2 of 4 telescopes, τ = 20 ns;
FEB coincidences from on-axis gamma rays, runs 110–130):

| Condition | Mean coincident FEBs | Topological factor | NSB stereo rate |
|---|---|---|---|
| Multiplicity only | – | 1 | 931 kHz |
| Topological, threshold 0 | 32.3 | 0.198 | 184 kHz |
| Topological, threshold 5% | 5.35 | 0.033 | 31 kHz |