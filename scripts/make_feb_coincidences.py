"""
Counts of the FEBs where the Cherenkov light of the same shower falls in each pair of telescopes,
used to build the look-up table of the topological stereo trigger (low_trigger.stereo.feb_lut).

For each event and telescope, the FEB of the pixel with the most true photoelectrons is taken
(pixel // 49, since the 49 pixels of each FEB are consecutive). For each ordered pair of telescopes
(t1, t2), counts[f1, f2] is the number of showers with their brightest pixel in FEB f1 of t1 and in
FEB f2 of t2. It should be run on diffuse gamma rays, to cover the whole camera.

Example:
python scripts/make_feb_coincidences.py -i "/path/gamma_diffuse_run{run}.r0.h5" -r 1 20 -o feb_coincidences.npz
"""
import argparse
from itertools import permutations

import numpy as np
import tables

N_FEB, PIX_PER_FEB = 163, 49


def brightest_febs(path, chunk=2000):
    """{(obs_id, event_id): {tel_id: feb}}: FEB of the brightest pixel of each telescope image."""
    out = {}
    with tables.open_file(path) as h5:
        for node in h5.get_node("/simulation/event/telescope/images"):
            tel_id = int(node._v_name.split("_")[1])
            for start in range(0, node.nrows, chunk):
              stop = min(start + chunk, node.nrows)
              images = node.read(start, stop, field="true_image")
              febs = images.argmax(axis=1) // PIX_PER_FEB
              has_light = images.sum(axis=1) > 0              # sin luz, argmax daría la FEB 0
              obs = node.read(start, stop, field="obs_id")
              evt = node.read(start, stop, field="event_id")
              for o, e, f, lit in zip(obs, evt, febs, has_light):
                  if lit:
                      out.setdefault((int(o), int(e)), {})[tel_id] = int(f)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Input file pattern with {run}")
    parser.add_argument("-r", "--runs", nargs=2, type=int, required=True, metavar=("FIRST", "LAST"))
    parser.add_argument("-o", "--output", required=True, help="Output .npz file")
    args = parser.parse_args()

    counts = {}
    n_events = 0
    for run in range(args.runs[0], args.runs[1] + 1):
        for febs in brightest_febs(args.input.format(run=run)).values():
            if len(febs) < 2:
                continue
            n_events += 1
            for t1, t2 in permutations(sorted(febs), 2):
                counts.setdefault(f"{t1}_{t2}", np.zeros((N_FEB, N_FEB), dtype=np.int64))[febs[t1], febs[t2]] += 1
        print(f"run {run}: {n_events} events with >= 2 telescopes so far")

    np.savez(args.output, **counts)
    print(f"Saved counts for telescope pairs {sorted(counts)} in {args.output}")


if __name__ == "__main__":
    main()
