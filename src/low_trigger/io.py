"""
Reading of the trigger files (get_l1tdscan.py) and of the CTLearn prediction files.
"""
import os

import numpy as np
import tables as pytables
from astropy.table import Table, join, vstack
from ctapipe.io import read_table

# Columns of the trigger files used in the analysis (the others are not read into memory)
TRIGGER_COLUMNS = [
    "obs_id", "event_id", "tel_id", "table_index",
    "true_image_sum", "true_energy",
    "low_trigger", "trigger_per_sample", "trigger_per_sector",
]
KEYS = ["obs_id", "event_id", "tel_id"]


def file_list(pattern, runs, check=True):
    """
    Build the list of files from a pattern with ``{run}`` and a list of runs,
    e.g. ``file_list("/path/run{run}/predict.h5", range(19, 26))``.
    With ``check`` it raises an error listing the files that do not exist.
    """
    files = [pattern.format(run=run) for run in runs]
    if check:
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))
    return files


def read_trigger_file(path, columns=TRIGGER_COLUMNS):
    """Read one trigger file, keeping only the available ``columns``."""
    table = read_table(path, "/table")
    table.keep_columns([c for c in columns if c in table.colnames])
    # 0/1 columns as booleans: 8 times less memory than int64
    for name in ("trigger_per_sample", "trigger_per_sector", "low_trigger", "triggered_febs"):
        if name in table.colnames:
            table[name] = np.asarray(table[name]).astype(bool)
    return table


def read_predictions(path, tel_ids=(1, 2, 3, 4), prefix="CTLearn"):
    """
    Read the telescope predictions of a CTLearn prediction file.

    Returns a table with ``obs_id``, ``event_id``, ``tel_id`` and ``cnn_prediction``
    (NaN when CTLearn marks the prediction as not valid).
    """
    with pytables.open_file(path, mode="r") as h5file:
        nodes = [f"/dl2/event/telescope/classification/{prefix}/tel_{tel_id:03d}" for tel_id in tel_ids]
        nodes = [node for node in nodes if node in h5file]
    if not nodes:
        raise ValueError(f"No telescope predictions found in '{path}'.")

    parts = []
    for node in nodes:
        t = read_table(path, node)
        prediction = np.asarray(t[f"{prefix}_tel_prediction"], dtype=float)
        if f"{prefix}_tel_is_valid" in t.colnames:
            prediction[~np.asarray(t[f"{prefix}_tel_is_valid"], dtype=bool)] = np.nan
        parts.append(Table({key: t[key] for key in KEYS} | {"cnn_prediction": prediction}))
    return vstack(parts)


def read_chain_table(trigger_files, prediction_files=None, extra_columns=()):
    """
    Read and stack the trigger files of a trigger chain.

    If ``prediction_files`` are given (one per trigger file, in the same order), the
    CNN prediction of each frame is added as ``cnn_prediction``, with NaN for the frames
    that the CNN did not evaluate. A ``file_index`` column tells the file of each row.
    """
    if prediction_files is not None and len(prediction_files) != len(trigger_files):
        raise ValueError(
            f"{len(trigger_files)} trigger files but {len(prediction_files)} prediction files."
        )

    tables_per_file = []
    for i, trigger_path in enumerate(trigger_files):
        table = read_trigger_file(trigger_path, columns=TRIGGER_COLUMNS + list(extra_columns))
        if prediction_files is not None:
            predictions = read_predictions(prediction_files[i])
            n_frames = len(table)
            table = join(table, predictions, keys=KEYS, join_type="left")
            if len(table) != n_frames:
                raise ValueError(f"Duplicated frames after joining '{prediction_files[i]}'.")
            # Every prediction must correspond to a frame of the trigger file
            matched = ~np.ma.getmaskarray(table["cnn_prediction"])
            if matched.sum() != len(predictions):
                raise ValueError(
                    f"{len(predictions) - matched.sum()} predictions of '{prediction_files[i]}' "
                    f"do not match any frame of '{trigger_path}' (wrong file pairing?)."
                )
            table["cnn_prediction"] = np.asarray(np.ma.filled(table["cnn_prediction"], np.nan), dtype=float)
        table["file_index"] = i
        tables_per_file.append(table)
    return vstack(tables_per_file)
