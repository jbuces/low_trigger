import numpy as np
import os
import sys
import argparse
from eventio import SimTelFile
import tables
from ctapipe.io import read_table
from astropy.table import Table, join, vstack, unique
import logging
from itertools import combinations
import matplotlib.pyplot as plt
import pickle

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger()


help_message = """
This script reads low level trigger tables and gets the stats for the stereo.

"""

def build_stereo_event_tables(table):
    table.sort(['file_index', 'obs_id', 'event_id', 'tel_id'])

    group_cols = ['obs_id', 'event_id', 'file_index', 'true_energy']
    grouped = table.group_by(group_cols)

    rows = []
    for key, group in zip(grouped.groups.keys, grouped.groups):
        tel_ids = group['tel_id'].tolist()
        low_trigger = group['low_trigger'].tolist()
        peak_sample = group['peak_sample'].tolist()
        true_image_sum = group['true_image_sum'].tolist()
        trigger_per_sample = group['trigger_per_sample'].tolist()
        max_pix = group['max_pix'].tolist()
        triggered_febs = group['triggered_febs'].tolist()

        rows.append({
            'file_index': key['file_index'],
            'obs_id': key['obs_id'],
            'event_id': key['event_id'],
            'tel_id': np.array(tel_ids),
            'true_energy' : key['true_energy'],
            'low_trigger_array': np.array(low_trigger),
            'peak_sample': np.array(peak_sample),
            'true_image_sum': np.array(true_image_sum),
            'trigger_per_sample': np.array(trigger_per_sample),
            'triggered_febs':np.array(triggered_febs),
            'max_pix': np.array(max_pix),
        })

    stereo_table = Table(rows)
    stereo_table.sort(['file_index', 'obs_id', 'event_id'])
    return stereo_table.copy()


def build_stereo_nsb_tables(table, n_tels=4, combos_needed=10, seed=123):

    np.random.seed(seed)

    run_tables = {idx: table[table['file_index'] == idx] for idx in np.unique(table['file_index'])}
    run_tables = [run_tables[idx] for idx in sorted(run_tables.keys())]

    n_runs = len(run_tables)
    if n_tels > n_runs:
        raise ValueError(f"No hay suficientes runs ({n_runs}) para combinar {n_tels} telescopios.")

    all_combos = list(combinations(range(n_runs), n_tels))
    if len(all_combos) < combos_needed:
        combos_needed = len(all_combos)

    selected = np.random.choice(len(all_combos), combos_needed, replace=False)
    selected_combos = [all_combos[i] for i in selected]

    stereo_tables = []
    for combo in selected_combos:
        tables = [run_tables[i] for i in combo]
        n_events = min(len(t) for t in tables)

        rows = []
        for i in range(n_events):
            rows.append({
                'combo_index': combo,
                'file_index': np.array(combo),
                'event_id': i,
                'tel_id': np.arange(1, n_tels + 1),
                'low_trigger_array': np.array([t['low_trigger'][i] for t in tables]),
                'trigger_per_sample': np.array([t['trigger_per_sample'][i] for t in tables]),
                'triggered_febs':  np.array([t['triggered_febs'][i] for t in tables]),
            })

        stereo_tables.append(Table(rows))

    return vstack(stereo_tables).copy()

def save_stereo_table(table, path):
    with open(path, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_stereo_table(path):
    with open(path, "rb") as f:
        tbl_loaded = pickle.load(f)
    return tbl_loaded

def stereo_simple_trigger(table, min_triggers=1):
    """
    Añade una columna 'low_trigger' a la tabla, que vale 1 si al menos `min_triggers`
    telescopios del 'low_trigger_array' tienen valor 1.
    """
    out_table = table.copy()

    low_trigger = []
    for arr in out_table['low_trigger_array']:
        low_trigger.append(1 if np.sum(arr) >= min_triggers else 0)
    
    out_table['time_low_trigger'] = low_trigger
    return out_table

def stereo_time_trigger(table, window=20, name='time_low_trigger'):
    out_table = table.copy()

    stereo_trigger = []
    time_trigger_arrays = []

    for trig, low_trig in zip(out_table['trigger_per_sample'],
                              out_table['low_trigger_array']):
        
        n_tel = trig.shape[0]

        # inicializamos salida con misma forma que low_trig
        time_array = np.zeros_like(low_trig)
        stereo_flag = 0
        for i in range(n_tel):

            if low_trig[i] != 1:
                continue

            t_i = np.where(trig[i] == 1)[0]
            if len(t_i) == 0:
                continue

            for j in range(i+1, n_tel):

                if low_trig[j] != 1:
                    continue

                t_j = np.where(trig[j] == 1)[0]
                if len(t_j) == 0:
                    continue

                diffs = np.abs(t_i[:,None] - t_j[None,:])

                if np.any(diffs <= window):
                    time_array[i] = 1
                    time_array[j] = 1
                    stereo_flag = 1  
        stereo_trigger.append(stereo_flag)
        time_trigger_arrays.append(time_array)


    out_table[name] = stereo_trigger
    out_table['time_trigger_tels'] = time_trigger_arrays

    return out_table


def plot_peak_samples_diff(stereo_table, logscale = True):
    tel_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    peak_diffs = {f"{a}-{b}": [] for a, b in tel_pairs}

    for row in stereo_table:
        tel_ids = row['tel_id']
        peaks = row['peak_sample']
        
        tel_to_peak = dict(zip(tel_ids, peaks))
        
        for (a, b) in tel_pairs:
            if a in tel_to_peak and b in tel_to_peak:
                diff = tel_to_peak[a] - tel_to_peak[b]
                peak_diffs[f"{a}-{b}"].append(diff)

    for key in peak_diffs:
        peak_diffs[key] = np.array(peak_diffs[key])
    for key, diffs in peak_diffs.items():
        if len(diffs) == 0:
            continue  

        plt.figure(figsize=(6, 4))
        plt.hist(diffs, bins=78, histtype='stepfilled', alpha=0.7)
        plt.title(f"Δ peak sample: tel {key.split('-')[0]} - tel {key.split('-')[1]}")
        plt.xlabel("Diferencia de peak sample")
        plt.ylabel("Frecuencia")
        if logscale:
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def topo_trigger(table, feb_coincidences):
    time_low_trigger  = table['time_low_trigger']
    time_trigger_tels = table['time_trigger_tels']   # shape (N, n_tels_event)
    triggered_febs    = table['triggered_febs']      # shape (N, n_tels_event, n_feb)
    tel_id            = table['tel_id']              # shape (N, n_tels_event)

    low_trigger = np.zeros(len(time_low_trigger), dtype=int)

    for evt in range(len(time_low_trigger)):

        # Si no hay low trigger global → saltar
        if time_low_trigger[evt] != 1:
            continue

        # Índices locales de telescopios que triggeraron
        active_idx = np.where(time_trigger_tels[evt] == 1)[0]

        if len(active_idx) < 2:
            continue

        # Para cada par de índices locales
        for i, idx1 in enumerate(active_idx):
            for idx2 in active_idx[i+1:]:

                # Telescopios reales (1..N)
                t1 = tel_id[evt][idx1]
                t2 = tel_id[evt][idx2]

                # FEBs activados en cada telescopio
                febs_t1 = np.where(triggered_febs[evt][idx1] == 1)[0]
                febs_t2 = np.where(triggered_febs[evt][idx2] == 1)[0]

                if len(febs_t1) == 0 or len(febs_t2) == 0:
                    continue

                # Comprobación de coincidencias FEB-FEB
                for f1 in febs_t1:
                    allowed_f2 = feb_coincidences[(t1, t2)][f1]

                    if any(f2 in allowed_f2 for f2 in febs_t2):
                        low_trigger[evt] = 1
                        break

                if low_trigger[evt] == 1:
                    break

            if low_trigger[evt] == 1:
                break

    out = table.copy()
    out['low_trigger'] = low_trigger
    return out

