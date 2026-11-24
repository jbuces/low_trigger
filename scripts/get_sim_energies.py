"""
Energy (GeV) of every simulated CORSIKA shower, one value per shower (not per reuse),
used as denominator of the trigger efficiencies (SimulatedShowers in low_trigger.chain).

It reads the EventHeader blocks (type 1202) of the CORSIKA files with eventio: there is one
EventHeader per shower and its 'total_energy' is in GeV. The raw EventIOFile reader is used
instead of IACTFile, which fails on unknown block types (e.g. 1213).

The output .npz has two arrays: 'energy' (GeV) and 'run' (run of each shower).

Run it where the CORSIKA files are (PIC). It only needs numpy and eventio:
    python3 -m venv ~/venv_eventio && source ~/venv_eventio/bin/activate
    pip install eventio numpy
    python get_sim_energies.py -i "/pnfs/pic.es/data/cta/LST/Data/UCM-temp/gamma/corsika_run{run}.corsika.gz" \
        -r 1 50 -o sim_energies.npz
"""
import argparse
import os
import sys

import numpy as np
from eventio import EventIOFile
from eventio.iact.objects import EventHeader


def shower_energies(path):
    """Energy (GeV) of each shower of one CORSIKA file."""
    energies = []
    with EventIOFile(path) as f:
        for obj in f:
            if isinstance(obj, EventHeader):
                energies.append(float(obj.parse()["total_energy"]))
    return np.array(energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="CORSIKA file pattern with {run}")
    parser.add_argument("-r", "--runs", nargs=2, type=int, required=True, metavar=("FIRST", "LAST"),
                        help="First and last run (both included)")
    parser.add_argument("-o", "--output", required=True, help="Output .npz file")
    parser.add_argument("-n", "--showers-per-run", type=int, default=10000,
                        help="Expected showers per file, to flag truncated files (default 10000)")
    args = parser.parse_args()

    files = {run: args.input.format(run=run) for run in range(args.runs[0], args.runs[1] + 1)}
    missing = [path for path in files.values() if not os.path.exists(path)]
    if missing:
        print("Missing files:", *missing, sep="\n  ")
        sys.exit(1)

    energies, runs, failed = [], [], []
    for run, path in files.items():
        try:
            e = shower_energies(path)
        except Exception as err:
            print(f"ERROR in {path}: {err!r}")
            failed.append(path)
            continue
        flag = "" if len(e) == args.showers_per_run else "  <-- CHECK"
        print(f"run {run}: {len(e)} showers{flag}")
        energies.append(e)
        runs.append(np.full(len(e), run, dtype=np.int32))

    energies, runs = np.concatenate(energies), np.concatenate(runs)
    np.savez(args.output, energy=energies, run=runs)
    print(f"\n{len(energies)} showers, E = {energies.min():.3g} - {energies.max():.3g} GeV, saved in {args.output}")
    if failed:
        print("Files with errors (not included):", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
