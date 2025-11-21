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
from ctapipe.io import read_table
from matplotlib.gridspec import GridSpec

from astropy.table import vstack, join, Column
from ctapipe.io import read_table

def get_camera_trigger_table(ctlearn_files_path, trigger_files):
    if isinstance(ctlearn_files_path, str):
        ctlearn_files_path = [ctlearn_files_path]
    if isinstance(trigger_files, str):
        trigger_files = [trigger_files]

    all_tables = []

    if len(ctlearn_files_path) != len(trigger_files):
        raise ValueError("ctlearn_files_path y trigger_files deben tener el mismo número de archivos")

    for j, (file_path, trig_path) in enumerate(zip(ctlearn_files_path, trigger_files)):

        # --- Leer tabla de triggers ---
        try:
            trigger_tab = read_table(trig_path, "/table")
        except Exception:
            print(f"⚠️ No se pudo leer /table en {trig_path}")
            continue

        # Seleccionar solo las columnas necesarias
        keep_cols = [
            "obs_id", "event_id", "tel_id", "true_image_sum",
            "true_energy", "trigger_per_sample", "low_trigger", "triggered_febs", "peak_sample", "max_pix"
        ]
        trigger_tab = trigger_tab[[c for c in keep_cols if c in trigger_tab.colnames]]

        cam_tabs = []

        # --- Leer las 4 cámaras CTLearn ---
        for i in range(1, 5):
            table_path = f"/dl2/event/telescope/classification/CTLearn/tel_{i:03d}"
            try:
                cam_tab = read_table(file_path, table_path)
            except Exception:
                print(f"⚠️ {table_path} no encontrado en {file_path}")
                continue

            # Seleccionar columnas relevantes (algunas pueden variar entre versiones)
            keep_cols_cam = [
                "obs_id", "event_id", "tel_id",
                "CTLearn_tel_prediction", "CTLearn_tel_is_valid"
            ]
            cam_tab = cam_tab[[c for c in keep_cols_cam if c in cam_tab.colnames]]

            cam_tabs.append(cam_tab)

        if len(cam_tabs) == 0:
            print(f"⚠️ No se encontraron cámaras válidas en {file_path}")
            continue

        # --- Apilar cámaras ---
        cam_all = vstack(cam_tabs)

        # --- Hacer LEFT JOIN (trigger como izquierda) ---
        merged_tab = join(
            trigger_tab,
            cam_all,
            keys=["obs_id", "event_id", "tel_id"],
            join_type="left",
        )

        # --- Añadir índice del archivo ---
        merged_tab.add_column(Column([j] * len(merged_tab), name="file_index"))

        all_tables.append(merged_tab)
    all_tables = vstack(all_tables)
    # --- Stack final de todos los archivos ---
    if all_tables:
        return all_tables
    else:
        return None




def get_camera_predictions(pred_table, nsb=False):

    if nsb == False:
        pe_bins = {
            "20-50": [],
            "50-100": [],
            "100-200": [],
            ">200": []
        }
        cherenkov_pe = np.array(pred_table["true_image_sum"]).flatten()

    preds = np.array(pred_table["CTLearn_tel_prediction"]).flatten()

    if nsb == False:
        pe_bins["20-50"].extend(preds[(cherenkov_pe >= 20) & (cherenkov_pe < 50)])
        pe_bins["50-100"].extend(preds[(cherenkov_pe >= 50) & (cherenkov_pe < 100)])
        pe_bins["100-200"].extend(preds[(cherenkov_pe >= 100) & (cherenkov_pe < 200)])
        pe_bins[">200"].extend(preds[cherenkov_pe >= 200])
    else:
        pe_bins = None

    return preds, pe_bins


def get_camera_prediction_plots(preds, pe_bins, nsb = False):
    plt.figure(figsize=(10, 6))
    
    lab = 'nsb' if nsb else 'shower'
    plt.hist(preds, bins=300, alpha=0.6, label=f'{lab}', histtype='step', linewidth=2, color='C1')
    plt.yscale('log')
    plt.xlabel('CTLearn_tel_prediction (nsbness)')
    plt.ylabel(f'Number of events')
    plt.title(f'Per event prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if nsb == False:
        plt.figure(figsize=(10, 6))
        colors = ['C0', 'C1', 'C2', 'C3']
        for label, color in zip(pe_bins.keys(), colors):
            plt.hist(pe_bins[label], bins=300, histtype='step', linewidth=2, label=f'PE {label}', color=color)
        plt.yscale('log')
        plt.xlabel('CTLearn_tel_prediction (nsbness)')
        plt.ylabel(f'Number of events')
        plt.title(f'PE binned per event prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def get_camera_nsb_threshold(per_event_table, freqs_kHz, ns_samples=31):

    # extraemos mínimos por evento
    min_preds = np.array(per_event_table["CTLearn_tel_prediction"])
    # definimos thresholds donde buscar
    thresholds = np.linspace(0, 1, 1000)

    # calculamos fracción de eventos activados para cada threshold

    triggered_fraction = [
        np.sum((min_preds < thr) & ~np.isnan(min_preds)) / len(min_preds)
        for thr in thresholds
    ]
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

def get_table_with_low_trigger(table, threshold):
    preds = np.array(table["CTLearn_tel_prediction"])
    out = table.copy()
    trigger = (preds <= threshold).astype(int)
    out["low_trigger"] = trigger
    return out


def plot_trigger_efficiency_camera(tables, names, thresholds, tdscans, get_ratio=None, pe_min=20, pe_max=250, pe_bin_width=10):
    # Define PE bins
    bins = np.arange(pe_min, pe_max + pe_bin_width, pe_bin_width)
    bin_centers = bins[:-1] + pe_bin_width / 2
    eff_list = []
    plt.figure(figsize=(11,7))
    for i, table in enumerate(tables):
        # Event PEs
        pes = np.array(table["true_image_sum"])
        if tdscans[i] == True:
            trigger_per_sample = np.array(table["trigger_per_sample"])
            tdscan_preds = np.any(trigger_per_sample[:, 10:31] == 1, axis=1)
        else:
            preds = np.array(table["CTLearn_tel_prediction"])

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
                else:
                    preds_in_bin = preds[mask]
                    valid_triggered = np.isfinite(preds_in_bin) & (preds_in_bin <= thresholds[i])
                    eff_val = np.sum(valid_triggered) / total
                    eff.append(eff_val)

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


def plot_trigger_efficiency_energy_camera(
    tables, names, tdscans, errors = True, stereo=False, effective_area=True, sim_rad=800, get_ratio=None, 
    e_min=5, e_max=5e4, e_bin_width=0.1, log_bins=True, ymin=0.01, colors = [0,1,2,3,4,5],line_styles = ['solid','solid','solid','solid','solid','solid'] 
):
    """
    Plot trigger efficiency vs true energy (in GeV).
    """
    plt.rcParams.update({
        "font.size": 16,       # tamaño base
        "axes.titlesize": 16,  # títulos de cada eje
        "axes.labelsize": 16,  # labels x/y
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    line_colors = [base_colors[c] for c in colors]
 
    if log_bins:
        bins = np.logspace(np.log10(e_min), np.log10(e_max), int(np.log10(e_max/e_min)/np.log10(1+e_bin_width)) + 1)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])  
    else:
        bins = np.arange(e_min, e_max + e_bin_width, e_bin_width)
        bin_centers = bins[:-1] + e_bin_width / 2

    eff_list = []
    err_list = [] 

    plt.figure(figsize=(11,7))

    for i, table in enumerate(tables):
        energies = np.array(table["true_energy"]) * 1e3   # En GeV
        if stereo is True or tdscans[i] is False:
            tdscan_preds = np.array(table["low_trigger"])
        else:
            trigger_per_sample = np.array(table["trigger_per_sample"])
            tdscan_preds = np.any(trigger_per_sample[:, 10:31] == 1, axis=1)
            

        eff = []
        err = []    

        for low, high in zip(bins[:-1], bins[1:]):
            mask = (energies >= low) & (energies < high)
            total = mask.sum()
            if total == 0:
                eff.append(0)
                err.append(0)
            else:
                eps = tdscan_preds[mask].sum() / total
                eff.append(tdscan_preds[mask].sum() / total)
                err.append(np.sqrt(eps*(1-eps)/total))

        eff_list.append(np.array(eff))
        err_list.append(np.array(err))

        if get_ratio is None:
            if errors:
                plt.errorbar(
                    bin_centers, eff, yerr=err,
                    color=line_colors[i], linestyle=line_styles[i],
                    label=f"{names[i]}", marker=''
                )
            else:
                plt.plot(
                    bin_centers, eff,
                    color=line_colors[i],
                    linestyle=line_styles[i],
                    label=f"{names[i]}"
                )
    if get_ratio is None:
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
    if get_ratio is not None and isinstance(get_ratio, int) and 0 <= get_ratio < len(eff_list):

        ref_eff = eff_list[get_ratio]

        # Figura con dos paneles SIN separación
        fig = plt.figure(figsize=(11, 8))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0)  # hspace=0 = pegados

        ax_eff = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_eff)

        # --------- Eficiencia arriba ---------
        for i, eff in enumerate(eff_list):
            if errors:
                ax_eff.errorbar(
                    bin_centers, eff, yerr=err_list[i],
                    color=line_colors[i], linestyle=line_styles[i],
                    label=f"{names[i]}", marker=''
                )

            else:
                ax_eff.plot(
                    bin_centers, eff,
                    color=line_colors[i],
                    linestyle=line_styles[i],
                    label=f"{names[i]}"
                )

        ax_eff.set_ylabel("Trigger efficiency")
        ax_eff.set_xscale('log')
        ax_eff.set_yscale('log')
        ax_eff.set_ylim(ymin, 1.05)
        ax_eff.grid(True, which="both")
        ax_eff.set_title("Trigger efficiency vs true energy")
        ax_eff.legend()

        # Ocultar etiquetas de X en el panel superior
        ax_eff.tick_params(labelbottom=False)

        # --------- Ratio abajo ---------
        for i, eff in enumerate(eff_list):
            ratio = np.divide(eff, ref_eff, out=np.zeros_like(eff), where=ref_eff != 0)
            ax_ratio.plot(
                bin_centers, ratio,
                color=line_colors[i],
                linestyle=line_styles[i],
                label=f"{names[i]} / {names[get_ratio]}"
            )
        ax_ratio.set_xlabel("True energy [GeV]")
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_xscale('log')
        ax_ratio.grid(True, which="both")

        plt.tight_layout()
        plt.show()

    if effective_area:
        plt.figure(figsize=(11,7))
        area_eff_list = []
        area_err_list = []

        A_sim = np.pi * sim_rad**2
        for i, eff in enumerate(eff_list):
            area_eff = eff * A_sim
            area_err = err_list[i] * A_sim
            area_eff_list.append(area_eff)
            area_err_list.append(area_err)

            if get_ratio is None:
                if errors:
                    plt.errorbar(bin_centers, area_eff, yerr=area_err, color=line_colors[i], linestyle=line_styles[i], label=f'{names[i]}')
                else:
                    plt.plot(bin_centers, area_eff, color=line_colors[i], linestyle=line_styles[i], label=f'{names[i]}')

        if get_ratio is None:
            plt.xlabel("True energy [GeV]")
            plt.ylabel("Effective area [m²]")
            plt.title("Effective area vs true energy")
            if log_bins:
                plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, which="both")
            plt.legend()
            plt.show()

        # Ratio de área efectiva respecto a una referencia
        if get_ratio is not None and 0 <= get_ratio < len(area_eff_list):
            ref_area = area_eff_list[get_ratio]
            
            fig = plt.figure(figsize=(11,8))
            gs = GridSpec(2,1, height_ratios=[3,1], hspace=0)
        
            ax_area = fig.add_subplot(gs[0])
            ax_ratio = fig.add_subplot(gs[1], sharex=ax_area)
        
            # Área efectiva
            for i, area_eff in enumerate(area_eff_list):
                if errors:
                    ax_area.errorbar(bin_centers, area_eff, yerr=area_err_list[i],
                        color=line_colors[i], linestyle=line_styles[i],
                        label=f'{names[i]}', marker='')
                else:
                    ax_area.plot(bin_centers, area_eff,
                        color=line_colors[i], linestyle=line_styles[i],
                        label=f'{names[i]}', marker='')
            ax_area.set_ylabel("Effective area [m²]")
            ax_area.set_xscale('log')
            ax_area.set_yscale('log')
            ax_area.grid(True, which="both")
            ax_area.legend()
            ax_area.set_title("Effective area vs true energy")
            ax_area.tick_params(labelbottom=False) 
        
            # Ratio
            for i, area_eff in enumerate(area_eff_list):
                ratio = np.divide(area_eff, ref_area, out=np.zeros_like(area_eff), where=ref_area != 0)
                ax_ratio.plot(bin_centers, ratio,
                              color=line_colors[i], linestyle=line_styles[i],
                              label=f'{names[i]} / {names[get_ratio]}')
            ax_ratio.set_xlabel("True energy [GeV]")
            ax_ratio.set_ylabel("A_eff ratio")
            ax_ratio.set_xscale('log')
            ax_ratio.grid(True, which="both")
            plt.tight_layout()
            plt.show()

