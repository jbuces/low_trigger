"""
get_stats.py

Module for extracting and combining statistics from TDSCAN L1 files
and CTLearn output files.
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
from astropy.table import Table, join, vstack
import tables
from pathlib import Path


def get_per_patch_prediction_table(ctlearn_files_path):
    if isinstance(ctlearn_files_path, str):
        ctlearn_files_path = [ctlearn_files_path]

    all_tables = []

    for file_path in ctlearn_files_path:
        with tables.open_file(file_path, mode='r') as f:
            for i in range(1, 5):
                table_path = f"/dl2/event/telescope/classification/CTLearn/tel_{i:03d}"
                if not f.__contains__(table_path):
                    print(f"{table_path} not available in {file_path}")
                    continue

                node = f.get_node(table_path)
                tab = Table({
                    "obs_id": node.col("obs_id"),
                    "event_id": node.col("event_id"),
                    "tel_id": node.col("tel_id"),
                    "prediction": node.col("CTLearn_tel_prediction"),
                    "patch_index": node.col("patch_index"),
                    "patch_class": node.col("patch_class"),
                    "cherenkov_pe": node.col("cherenkov_pe"),
                })
                all_tables.append(tab)

    if all_tables:
        return vstack(all_tables)
    else:
        return Table()

def get_predictions(pred_table, apply_tdscan = False, nsb=False):
    preds_class_0 = []
    preds_class_1 = []
    if nsb == False:
        pe_bins = {
            "0": [],
            "0–10": [],
            "10–20": [],
            "20–50": [],
            "50–200": [],
            ">200": []
        }
        cherenkov_pe = np.array(pred_table["cherenkov_pe"]).flatten()

    if len(pred_table) == 0:
        return preds_class_0, preds_class_1, pe_bins

    preds = np.array(pred_table["prediction"]).flatten()
    if apply_tdscan:
        mask = np.where(pred_table["low_trigger"]==1)[0]
        preds = preds[mask]

    if "patch_class" in pred_table.colnames:
        patch_classes = np.array(pred_table["patch_class"]).flatten()
        preds_class_0.extend(preds[patch_classes == 0])
        preds_class_1.extend(preds[patch_classes == 1])
    else:
        preds_class_0 = None
        preds_class_1 = preds
    if preds_class_0 != None:
        pe_bins["0"].extend(preds[cherenkov_pe == 0])
        pe_bins["0–10"].extend(preds[(cherenkov_pe > 0) & (cherenkov_pe < 10)])
        pe_bins["10–20"].extend(preds[(cherenkov_pe >= 10) & (cherenkov_pe < 20)])
    if nsb == False:
        pe_bins["20–50"].extend(preds[(cherenkov_pe >= 20) & (cherenkov_pe < 50)])
        pe_bins["50–200"].extend(preds[(cherenkov_pe >= 50) & (cherenkov_pe < 200)])
        pe_bins[">200"].extend(preds[cherenkov_pe >= 200])
    else:
        pe_bins = None

    return preds_class_0, preds_class_1, pe_bins


def get_prediction_plots(preds_class_0, preds_class_1, pe_bins, nsb = False):
    # ---- Plot 1: Class histograms ----
    plt.figure(figsize=(10, 6))
    if preds_class_0:
        plt.hist(preds_class_0, bins=300, alpha=0.6, label='NSB', histtype='step', linewidth=2, color='C0')
        title = "patch"
        titles = "patches"
    else:
        title = "event"
        titles = "events"
    lab = 'nsb' if nsb else 'shower'
    plt.hist(preds_class_1, bins=300, alpha=0.6, label=f'{lab}', histtype='step', linewidth=2, color='C1')
    plt.yscale('log')
    plt.xlabel('CTLearn_tel_prediction (nsbness)')
    plt.ylabel(f'Number of {titles}')
    plt.title(f'Per {title} prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if nsb == False:
        # ---- Plot 2: PE bin histograms ----
        plt.figure(figsize=(10, 6))
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C6']
        for label, color in zip(pe_bins.keys(), colors):
            plt.hist(pe_bins[label], bins=300, histtype='step', linewidth=2, label=f'PE {label}', color=color)
        plt.yscale('log')
        plt.xlabel('CTLearn_tel_prediction (nsbness)')
        plt.ylabel(f'Number of {titles}')
        plt.title(f'PE binned per {title} prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def get_ctlearn_threshold_per_patch(nsb_efficiencies, preds_class_0, preds_class_1):
    all_preds = np.concatenate([preds_class_0, preds_class_1])
    all_classes = np.concatenate([np.zeros_like(preds_class_0), np.ones_like(preds_class_1)])

    # Thresholds to scan
    thresholds = np.linspace(0, 1, 100)

    eff_nsb = []
    eff_cosmic = []
    purity_nsb = []
    purity_cosmic = []

    for t in thresholds:
        triggered_nsb = (preds_class_0 < t).sum()
        triggered_cosmic = (preds_class_1 < t).sum()

        total_triggered = triggered_nsb + triggered_cosmic

        eff_nsb.append(triggered_nsb / len(preds_class_0))
        eff_cosmic.append(triggered_cosmic / len(preds_class_1))

        if total_triggered > 0:
            purity_nsb.append(triggered_nsb / total_triggered)
            purity_cosmic.append(triggered_cosmic / total_triggered)
        else:
            purity_nsb.append(0)
            purity_cosmic.append(0)

    eff_nsb = np.array(eff_nsb)
    thresholds = np.array(thresholds)

    ths = []
    for target in nsb_efficiencies:
        idx = np.argmin(np.abs(eff_nsb - target))
        print(f"NSB Efficiency ~ {target*100:.0f}% at threshold ≈ {thresholds[idx]:.4f} "
            f"(actual eff = {eff_nsb[idx]*100:.2f}%)")
        ths.append(thresholds[idx])

    plt.figure(figsize=(8,5))
    plt.plot(thresholds, eff_nsb, label='NSB trigger efficiency')
    plt.plot(thresholds, eff_cosmic, label='Cosmic trigger efficiency')
    for vertical in ths:
        plt.axvline(x=vertical, linestyle='--')
    plt.xlabel('CTLearn_tel_prediction (nsbness)')
    plt.ylabel('Triggered / Total (per class)')
    plt.title('Trigger efficiency per class')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def get_ctlearn_per_event_table(ctlearn_file_paths, nsb=False):
    # Aceptar string o lista
    if isinstance(ctlearn_file_paths, (str, Path)):
        ctlearn_file_paths = [ctlearn_file_paths]

    all_files_tables = []

    for ctlearn_file_path in ctlearn_file_paths:
        with tables.open_file(ctlearn_file_path, mode="r") as f:
            all_tel_tables = []
            energy_path = "/simulation/event/subarray/shower"
            energy_table = f.get_node(energy_path)
            energy_tab = Table({
                "obs_id": energy_table.col("obs_id"),
                "event_id": energy_table.col("event_id"),
                "true_energy": energy_table.col("true_energy")
            })
            for i in range(1, 5):
                # problemas cn los files de nsb y no tienen cherenkov_pe asi que
                if nsb == False:
                    # ---- Imagenes ----
                    image_path = f"/simulation/event/telescope/images/tel_{i:03d}"
                    if not f.__contains__(image_path):
                        print(f"{image_path} not available in {ctlearn_file_path}")
                        continue

                    image_table = f.get_node(image_path)
                    img_tab = Table({
                        "obs_id": image_table.col("obs_id"),
                        "event_id": image_table.col("event_id"),
                        "tel_id": image_table.col("tel_id"),
                        "cherenkov_pe": image_table.col("true_image_sum")
                    })

                # ---- Predicciones ----
                pred_path = f"/dl2/event/telescope/classification/CTLearn/tel_{i:03d}"
                if not f.__contains__(pred_path):
                    print(f"{pred_path} not available in {ctlearn_file_path}")
                    continue

                pred_table = f.get_node(pred_path)

                if "low_trigger" in pred_table.colnames:
                    low_trigger_col = pred_table.col("low_trigger")
                else:
                    low_trigger_col = np.zeros(len(pred_table), dtype=int)  # poner ceros si no existe
                    print("La tabla no posee información sobre el tdscan trigger")

                pred_tab = Table({
                    "obs_id": pred_table.col("obs_id"),
                    "event_id": pred_table.col("event_id"),
                    "tel_id": pred_table.col("tel_id"),
                    "prediction": pred_table.col("CTLearn_tel_prediction"),
                    "low_trigger": low_trigger_col,
                })

                # Agrupar predicciones → 1 fila por evento
                grouped = pred_tab.group_by(["obs_id", "event_id", "tel_id"])
                rows = []
                for key, group in zip(grouped.groups.keys, grouped.groups):
                    obs_id, event_id, tel_id = key
                    preds = list(group["prediction"])
                    low = group["low_trigger"][0]  # todos iguales → tomo el primero
                    rows.append((obs_id, event_id, tel_id, preds, low, min(preds)))

                pred_per_event = Table(
                    rows=rows,
                    names=["obs_id", "event_id", "tel_id", "predictions", "low_trigger", "prediction"]
                )
            
                if nsb == False:
                # ---- Merge imagen + predicciones ----
                    pred_per_event = join(
                        img_tab, pred_per_event,
                        keys=["obs_id", "event_id", "tel_id"],
                        join_type="inner"
                    )
                pred_per_event = join(
                    left=pred_per_event,
                    right=energy_tab,
                    keys=["obs_id", "event_id"],
                )
                pred_per_event.sort(['obs_id', 'event_id', 'tel_id'])

                # Raise errors in case of a corrupted file
                if np.any(np.isnan(pred_per_event["predictions"])) or np.any(np.isnan(pred_per_event["prediction"])):
                    raise ValueError("NaN detected in predictions or prediction column")

                all_tel_tables.append(pred_per_event)

            # Concatenar telescopios de este archivo
            if all_tel_tables:
                result = vstack(all_tel_tables)
                all_files_tables.append(result)

    # Concatenar archivos
    if all_files_tables:
        final_result = vstack(all_files_tables)
    else:
        final_result = Table()

    return final_result



def get_ctlearn_nsb_threshold(per_event_table, freqs_kHz, apply_tdscan = False, ns_samples=30):

    # extraemos mínimos por evento
    min_preds = np.array(per_event_table["prediction"])
    # definimos thresholds donde buscar
    thresholds = np.linspace(0, 1, 500)

    # calculamos fracción de eventos activados para cada threshold
    if apply_tdscan:
        mask_triggered = np.array(per_event_table["low_trigger"]==1)
        trig_preds = min_preds[mask_triggered] 

        triggered_fraction = [(trig_preds < thr).sum()/len(min_preds) for thr in thresholds]
        triggered_fraction = np.array(triggered_fraction)
    else:
        triggered_fraction = [(min_preds < thr).mean() for thr in thresholds]
        triggered_fraction = np.array(triggered_fraction)

    target_fractions = [f * ns_samples * 1e-6 for f in freqs_kHz]
    thr_values = []
    for i, frac in enumerate(target_fractions):
        idx = np.searchsorted(triggered_fraction, frac)

        thr_value = thresholds[idx]
        thr_values.append(thr_value)
        print(f"≈ Threshold {thr_value:.3f} da un {frac *100:.3f}% de eventos disparados, es decir {freqs_kHz[i]} kHz")
    
        # plot
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, triggered_fraction, lw=2)
    for i, thr in enumerate(thr_values):
        plt.axvline(x=thr, linestyle='--', label=f"{freqs_kHz[i]} kHz")
    plt.xlabel("CTLearn Threshold")
    plt.ylabel("Triggered fraction per event")
    plt.title("Trigger curve (min_prediction < threshold)")
    plt.grid(True)
    plt.show()

    return thr_values

def plot_trigger_efficiency(tables, names, thresholds, tdscans, apply_tdscan, get_ratio=None, pe_min=20, pe_max=250, pe_bin_width=10):
    # Define PE bins
    bins = np.arange(pe_min, pe_max + pe_bin_width, pe_bin_width)
    bin_centers = bins[:-1] + pe_bin_width / 2
    eff_list = []
    plt.figure(figsize=(11,7))
    for i, table in enumerate(tables):
        # Event PEs
        pes = np.array(table["cherenkov_pe"])
        if tdscans[i] == True:
            tdscan_preds = np.array(table["low_trigger"])
        elif apply_tdscan[i]==True:
            tdscan_preds = np.array(table["low_trigger"])
            preds = np.array(table["prediction"])
            triggered = preds <= thresholds[i]
        else:
            preds = np.array(table["prediction"])
            triggered = preds <= thresholds[i]

        # efficiency per bin
        eff = []

        for low, high in zip(bins[:-1], bins[1:]):
            mask = (pes >= low) & (pes < high)
            total = mask.sum()
            if total == 0:
                eff.append(0)
            else:
                if tdscans[i]:
                    eff.append(tdscan_preds[mask].sum() / total)
                elif apply_tdscan[i]:
                    eff.append(((triggered & (tdscan_preds == 1))[mask]).sum() / total)
                else:
                    eff.append(triggered[mask].sum() / total)
            
        eff = np.array(eff)
        eff_list.append(eff)

        plt.plot(bin_centers, eff, marker='', label=f'{names[i]}')
        plt.xlabel("Total pe per event")
        plt.ylabel("Trigger efficiency")
        plt.title(f"Trigger efficiency vs pe")
        plt.yscale('log')
        
        plt.grid(True)

    plt.legend()
    plt.show()

    if get_ratio is not None and isinstance(get_ratio, int) and 0 <= get_ratio < len(eff_list):
        ref_eff = eff_list[get_ratio]
        plt.figure(figsize=(11,7))
        for i, eff in enumerate(eff_list):
            # if i == get_ratio:
            #     continue  # no hacemos ratio con la misma
            ratio = np.divide(eff, ref_eff, out=np.zeros_like(eff), where=ref_eff!=0)
            plt.plot(bin_centers, ratio, label=f'{names[i]} / {names[get_ratio]}')

        plt.xlabel("Total pe per event")
        plt.ylabel("Efficiency ratio")
        plt.title(f"Efficiency ratio (reference: {names[get_ratio]})")
        plt.grid(True)
        plt.legend()
        plt.show()

    return bin_centers, eff

def plot_trigger_efficiency_energy(tables, names, thresholds, tdscans, apply_tdscan, get_ratio=None, e_min=10, e_max=1e4, e_bin_width=0.1, log_bins=True, ymin=0.01):
    """
    Plot trigger efficiency vs true energy (in TeV).
    """

    # Definir bins de energía (logarítmicos o lineales)
    if log_bins:
        bins = np.logspace(np.log10(e_min), np.log10(e_max), int(np.log10(e_max/e_min)/np.log10(1+e_bin_width)) + 1)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])  
    else:
        bins = np.arange(e_min, e_max + e_bin_width, e_bin_width)
        bin_centers = bins[:-1] + e_bin_width / 2

    eff_list = []
    plt.figure(figsize=(11,7))

    for i, table in enumerate(tables):
        energies = np.array(table["true_energy"]) * 1e3   # En GeV
        if tdscans[i]:
            tdscan_preds = np.array(table["low_trigger"])
        elif apply_tdscan[i]:
            tdscan_preds = np.array(table["low_trigger"])
            preds = np.array(table["prediction"])
            triggered = preds <= thresholds[i]
        else:
            preds = np.array(table["prediction"])
            triggered = preds <= thresholds[i]

        eff = []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (energies >= low) & (energies < high)
            total = mask.sum()
            if total == 0:
                eff.append(0)
            else:
                if tdscans[i]:
                    eff.append(tdscan_preds[mask].sum() / total)
                elif apply_tdscan[i]:
                    eff.append(((triggered & (tdscan_preds == 1))[mask]).sum() / total)
                else:
                    eff.append(triggered[mask].sum() / total)

        eff = np.array(eff)
        eff_list.append(eff)

        plt.plot(bin_centers, eff, marker='', label=f'{names[i]}')

    plt.xlabel("True energy [GeV]")
    plt.ylabel("Trigger efficiency")
    plt.title("Trigger efficiency vs True energy")
    if log_bins:
        plt.xscale('log')
    plt.ylim(ymin, 1.05)  
    plt.yscale('log')
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    # Ratio respecto a una referencia
    if get_ratio is not None and isinstance(get_ratio, int) and 0 <= get_ratio < len(eff_list):
        ref_eff = eff_list[get_ratio]
        plt.figure(figsize=(11,7))
        for i, eff in enumerate(eff_list):
            ratio = np.divide(eff, ref_eff, out=np.zeros_like(eff), where=ref_eff != 0)
            plt.plot(bin_centers, ratio, label=f'{names[i]} / {names[get_ratio]}')

        plt.xlabel("True energy [TeV]")
        plt.ylabel("Efficiency ratio")
        plt.title(f"Efficiency ratio (reference: {names[get_ratio]})")
        if log_bins:
            plt.xscale('log')
        plt.grid(True, which="both")
        plt.legend()
        plt.show()

    return bin_centers, eff



def get_merged_trigger_tables(files, path="/table", columns = ['table_index', 'obs_id', 'event_id',	'tel_id', 'true_image_sum', 'low_trigger', 'trigger_per_sample', 'true_energy']):
    all_tables = []

    for file_index, file_path in enumerate(files):
        with tables.open_file(file_path, mode='r') as f:
            if not f.__contains__("/table"):
                print(f"/table not found in {file_path}")
                continue

            node = f.get_node("/table")

            # Extract columns
            data = {col: node.col(col) for col in columns}

            # Add file index column
            data["file_index"] = np.full(len(node), file_index, dtype=np.int32)

            # Build your Table object (same structure as you already use)
            tab = Table(data)
            all_tables.append(tab)

    if not all_tables:
        raise ValueError("No valid /table found in the provided files.")

    # Stack them vertically
    stacked = vstack(all_tables)
    return stacked

def get_l1_tdscan_output_freq(tables, titles):
    models = []
    out_rates = []
    n_nsb_events = []

    for i, table in enumerate(tables):
        lt = table["low_trigger"]
        eff = np.sum(lt) / len(lt)
        out_freq = eff / 30 * 10**6  # en kHz
        n_nsb = len(lt)

        models.append(titles[i])
        out_rates.append(round(out_freq, 2))
        n_nsb_events.append(n_nsb)
        print(f"For the {titles[i]}, on {len(lt)} nsb events, we get an output frequency from accidental triggers of {out_freq} kHz")

    result_table = Table(
        [models, out_rates, n_nsb_events],
        names=("Model", "Output Rate (kHz)", "Number of NSB events")
    )

    return result_table