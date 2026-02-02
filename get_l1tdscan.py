import numpy as np
import os
import sys
import argparse
import tables
from ctapipe.io import read_table
from astropy.table import Table, join, vstack
import logging
from numpy.lib.stride_tricks import sliding_window_view


help_message = """
This script reads an h5 file produced by ctapipe from simtel, and applies different low level trigger algorithms.
Example usage:
nohup python3 get_l1tdscan_stats.py -i /lustre/ific.uv.es/ml/ucm147/datasets/TriggerlessData/nsb/NSB268MHZ/corsika_run0.r0.h5 -o /lhome/ext/ucm147/ucm1473/lowtrigger_stats/nsb/corsika_run0_2161cut_tdscan7.h5 -nf /lhome/ext/ucm147/ucm1473/dl1-data-handler/dl1_data_handler/ressources/newcam_sipm_patches.h5 -pts 7 -l1 2161 -epsl1 1 -trig tdscan -epst 1 > /lhome/ext/ucm147/ucm1473/lowtrigger_stats/nsb/corsika_run0_2161cut_tdscan7.log 2>&1 &
"""

def parse_args():
    parser = argparse.ArgumentParser(description=help_message)
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-add', '--add_mask', help='Boolean to add the mask to the table', required=True, default=True)
    parser.add_argument('-nf', '--neighbors_file', help='Neighbors file for the l1 and tdscan', required=True)
    parser.add_argument('-qstep', '--quantisation_step', help='For quantised modes gives the quantisation step', required=True, type=int)
    parser.add_argument('-l1', '--l1_threshold', help='Threshold for the l1 trigger', required=True, type=int)
    parser.add_argument('-epsl1', '--epsl1', help='Use flower (1), superflower (2), or FEB (3) for digital sum', required=True, type=int)
    parser.add_argument('-trig', '--trigger_type', help='``l1``,``rolling_l1``, ``tdscan``, ``tdscan_per_flower``,``quantised_l1``, ``quantised_rolled_digital_sum``, ``quantised_per_feb``', required=True, type=str, default="tdscan")
    parser.add_argument('-epst', '--epst', help='Frames before and after for tdscan', required=True, type=int)
    parser.add_argument('-pts', '--min_pts', help='Min pts for tdscan flower convolution, if flower_quantised threshold for the sum per FEB.', required=True, type=int)
    return parser.parse_args()

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    # Consola
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    # Archivo
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger

def main():
    args = parse_args()
    h5_file = args.input
    out_table = args.output
    epsl1 = args.epsl1
    l1_threshold = args.l1_threshold
    neighbors_file = args.neighbors_file
    eps_t = args.epst
    min_pts = args.min_pts
    trigger_type = args.trigger_type
    add_mask = args.add_mask
    q_step = args.quantisation_step

    output_dir = os.path.dirname(out_table)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    log_file = os.path.join(
        output_dir,
        os.path.basename(out_table).replace(".h5", ".log")
    )

    logger = setup_logger(log_file)

    logger.info("====================================")
    logger.info("Starting trigger simulation")
    logger.info(f"Input file     : {h5_file}")
    logger.info(f"Output file    : {out_table}")
    logger.info(f"Trigger type   : {trigger_type}")
    logger.info(f"L1 threshold   : {l1_threshold}")
    logger.info(f"EPSL1          : {epsl1}")
    logger.info(f"Min pts        : {min_pts}")
    logger.info("====================================")

    if q_step == 0:
        q_step = None
    if trigger_type == 'quantised_per_feb' and epsl1 != 3:
        raise ValueError('The quantised per feb trigger needs feb region size for the digital sum')
    
    l1_array, tdscan_list, feb_indices, cam_neighbor_array = get_neighbors_info(neighbors_file, epsl1)
    if epsl1==2 or epsl1==3:
        l1_mask = (l1_array >= 0).astype(int)
    else:
        l1_mask = None
    trigtable = get_table(h5_file)
    n_events = len(trigtable)
    l1_trigger, trig_per_sample, peak_samples, max_pixs, febs = [], [], [], [], []
    ampmax_nb_amplitudes, feb_amp_dists = [], []
    logger.info("Processing the data...")

    packed_masks = None
    n_pixels = None
    n_times = None
    bits = None
    packed_len = None
    bitorder = "big"

    # Iterate over all the events of the file
    with tables.open_file(h5_file, 'r') as h5file:
        for i, row in enumerate(trigtable):
            tel_id = row["tel_id"]
            table_idx = row["table_index"]
            tel_path = f'/r0/event/telescope/tel_{tel_id:03d}'
            sim_path = f'/simulation/event/telescope/images/tel_{tel_id:03d}'
            wf = h5file.get_node(tel_path)[table_idx]['waveform'][0,:,:]
            true_im = np.array(h5file.get_node(sim_path)[table_idx]['true_image'], dtype=int).reshape(-1, 1)
            max_pix = np.argmax(true_im)
            # Apply the trigger
            if trigger_type in ['quantised_l1', 'quantised_rolled_digital_sum', 'quantised_per_feb']:
                trig, trig_p_sample, bin_flowers, peak_sample, febs_on, ampmax, feb_dist  = apply_low_trigger(wf, l1_array, l1_threshold, tdscan_list, eps_t, min_pts, trigger_type, l1_mask, max_pix, feb_indices, q_step)
            else:
                trig, trig_p_sample, bin_flowers, peak_sample, febs_on  = apply_low_trigger(wf, l1_array, l1_threshold, tdscan_list, eps_t, min_pts, trigger_type, l1_mask, max_pix, feb_indices, q_step)

            l1_trigger.append(trig)
            trig_per_sample.append(trig_p_sample)
            peak_samples.append(peak_sample)
            max_pixs.append(max_pix)
            febs.append(febs_on)
            if trigger_type in ['quantised_l1', 'quantised_rolled_digital_sum', 'quantised_per_feb']:
                ampmax_nb_amplitudes.append(ampmax)
                feb_amp_dists.append(feb_dist)

            if add_mask == "True":
                bf = np.asarray(bin_flowers, dtype=np.uint8)  # shape (n_pixels, n_times)
                # En la primera iteración conocemos las dimensiones y preallocamos packed_masks
                if packed_masks is None:
                    n_pixels, n_times = bf.shape
                    bits = n_pixels * n_times
                    packed_len = (bits + 7) // 8
                    packed_masks = np.zeros((n_events, packed_len), dtype=np.uint8)

                # Flatten y packbits
                flat = bf.ravel(order="C")        # 0/1 flat
                p = np.packbits(flat, bitorder=bitorder)
                # Padding por seguridad
                if p.size < packed_len:
                    temp = np.zeros(packed_len, dtype=np.uint8)
                    temp[: p.size] = p
                    p = temp
                elif p.size > packed_len:
                    # Esto no debería pasar, pero por seguridad truncar
                    p = p[:packed_len]
                packed_masks[i, :] = p

            if i % 1000 == 0:
                logger.info(f"{i}/{n_events} events processed.")

    trigtable.add_column(l1_trigger, name="low_trigger")
    trigtable.add_column(trig_per_sample, name="trigger_per_sample")
    trigtable.add_column(peak_samples, name="peak_sample")
    trigtable.add_column(max_pixs, name="max_pix")
    trigtable.add_column(febs, name="triggered_febs")
    if trigger_type in ['quantised_l1', 'quantised_rolled_digital_sum', 'quantised_per_feb']:
        trigtable.add_column(ampmax_nb_amplitudes, name="max_feb")
        trigtable.add_column(feb_amp_dists, name="feb_amplitude_distribution")
   
    if add_mask=="True":
        if packed_masks is None:
            raise RuntimeError("add_mask=True pero no se generó ningún mask (packed_masks is None).")
        # Añadimos la columna empaquetada (uint8 2D)
        trigtable.add_column(packed_masks, name="trigger_mask")
    else:
        logger.info("add_mask not set to True: no trigger_mask column will be added.")

    trigtable.write(out_table, path='/table', format='hdf5', overwrite=True)

    if add_mask == "True":
        try:
            with tables.open_file(out_table, mode='a') as hf:
                if hasattr(hf.root, "table"):
                    node = hf.root.table
                else:
                    # por si la estructura cambia, intentar encontrar el node '/table'
                    node = hf.get_node('/table')
                node._v_attrs.trigger_mask_shape = np.array([int(n_pixels), int(n_times)], dtype=np.int32)
                node._v_attrs.trigger_mask_bits = int(bits)
                node._v_attrs.trigger_mask_packed_len = int(packed_len)
                node._v_attrs.trigger_mask_bitorder = bitorder
                logger.info("Wrote trigger_mask attributes to HDF5 (shape, bits, packed_len, bitorder).")
        except Exception as e:
            logger.warning(f"Failed to write HDF5 attributes for trigger_mask: {e}")
            
    logger.info("Trigger finished")
    
def get_neighbors_info(neighbors_file, epsl1):
    logger = logging.getLogger()
    logger.info("Getting the neighbors information")
    with tables.open_file(neighbors_file, mode="r") as f:
        cam_neighbor_array = f.root.neighbors.camera_neighbors[:]
        fl_neighbor_array_tdscan = f.root.neighbors.flower_neighbors_tdscan[:]
        fl_neighbor_array_l1 = f.root.neighbors.flower_neighbors_l1[:]
        supfl_neighbor_array_l1 = f.root.neighbors.superflower_neighbors_l1[:]
        feb_neighbors = f.root.neighbors.feb_neighbors[:]
        feb_indices = f.root.modules.indices[:]
        # Remove -1 padding from each row
        neighbor_tdscan_eps1_list = [row[row != -1].tolist() for row in fl_neighbor_array_tdscan]
        fl_neighbor_l1_list = [row[row != -1].tolist() for row in fl_neighbor_array_l1]
        supfl_neighbor_l1_list = [row[row != -1].tolist() for row in supfl_neighbor_array_l1]
        
        # neighbor_tdscan_eps1_list = []

        # with open("/lhome/ext/ucm147/ucm1473/dl1-data-handler/dl1_data_handler/ressources/neighbors_fpga0_no_overlap_border_1.txt", "r") as f:
        #     for linea in f:
        #         nums = linea.strip().split(",")     # separar por comas
        #         nums = [int(x) for x in nums if x]  # convertir a int (evita strings vacíos)
        #         neighbor_tdscan_eps1_list.append(nums)

        logger.info(f"longitud de lista tdscan {len(neighbor_tdscan_eps1_list)}")     # debería ser 1141
        logger.info(f"longitud de primera linea {len(neighbor_tdscan_eps1_list[0])}")       # primera fila
        logger.info(f"primera linea: {neighbor_tdscan_eps1_list[0]}")
        
    if epsl1==1:
        l1_list=fl_neighbor_array_l1
    elif epsl1==2:
        l1_list=supfl_neighbor_array_l1
    else:
        l1_list=feb_indices
        neighbor_tdscan_eps1_list= feb_neighbors

    return l1_list, neighbor_tdscan_eps1_list, feb_indices, cam_neighbor_array


def rolling_digital_sum(flower_sums, eps_t):
    arr_rolled = sliding_window_view(flower_sums, window_shape=eps_t, axis=1)
    flower_sums = arr_rolled.sum(axis=2)
    return flower_sums

def digital_sum(x, l1_list, l1_mask):
    # Assign the l1 neighbor list, summing on flowers, superflowers  or FEBs.
    if l1_mask is not None:
        vals = x[l1_list] 
        flower_sums = (vals * l1_mask[..., None]).sum(axis=1)
    else:
        flower_sums = x[np.array(l1_list)].sum(axis=1)
    return flower_sums

def quantised_digital_sum(flower_sums, l1_threshold, mask, quant_step):
     # sumar todas las flores de una misma feb
    flower_sums_red = np.maximum(flower_sums.astype(np.int32) - l1_threshold, 0)
    feb_sums = flower_sums_red.reshape(163, 7, 40)
    feb_summed = feb_sums.sum(axis=1)
    # cuantizar las amplitudes
    vals_q = np.floor_divide(feb_summed, quant_step)
    vals_q = np.clip(vals_q, 0, 127).astype(np.uint16)
    # numero de triggers por feb por sample
    mask_feb = mask.reshape(163, 7, 40)
    mask_feb_summed = mask_feb.sum(axis=1)
    waveform = np.stack([mask_feb_summed, vals_q], axis=-1)

    # para el maximo del evento de numero de flores
    values_amp = waveform[:,:,1]  
    ampmax = np.max(values_amp)

    return waveform, ampmax

def quantised_per_feb_digital_sum(feb_sums, l1_threshold, quant_step):
    vals_thr = np.maximum(feb_sums.astype(np.int32) - l1_threshold, 0)
    vals_q = np.floor_divide(vals_thr, quant_step)
    vals_q = np.clip(vals_q, 0, 1023).astype(np.uint16)
    # para el maximo del evento de numero de flores 
    ampmax = np.max(vals_q[:,:])
    return vals_q, ampmax

def quantised_rolled_digital_sum(flower_sums, l1_threshold, mask):
     # sumar todas las flores de una misma feb
    flower_sums_red = np.maximum(flower_sums.astype(np.int32) - l1_threshold, 0)
    feb_sums = flower_sums_red.reshape(163, 7, 40)
    feb_summed = feb_sums.sum(axis=1)
    # cuantizar las amplitudes
    arr_rolled = sliding_window_view(feb_summed, window_shape=3, axis=1)
    feb_rolled = arr_rolled.sum(axis=2)
    vals_q = np.floor_divide(feb_rolled, 3)
    vals_q = np.clip(vals_q, 0, 127).astype(np.uint16)
    vals_q = np.pad(vals_q, pad_width=((0, 0), (1, 1)), mode='constant', constant_values=0)

    # numero de triggers por feb por sample
    mask_feb = mask.reshape(163, 7, 40)
    mask_feb_summed = mask_feb.sum(axis=1)
    mask_arr_rolled = sliding_window_view(mask_feb_summed, window_shape=3, axis=1)
    maskfeb_rolled = mask_arr_rolled.sum(axis=2)
    maskfeb_rolled = np.pad(maskfeb_rolled, pad_width=((0, 0), (1, 1)), mode='constant', constant_values=0)

    waveform = np.stack([maskfeb_rolled, vals_q], axis=-1)

    # para el maximo del evento de numero de flores
    values_amp = waveform[:,:,1]  
    ampmax = np.max(values_amp)

    return waveform, ampmax


def tdscan_per_flower(bin_flowers, eps_t, min_pts):
    w_shape = 2 * eps_t + 1
    windows = sliding_window_view(bin_flowers, window_shape=w_shape, axis=1)
    sums = windows.sum(axis=2)
    thresholded = (sums >= min_pts).astype(int)
    padded = np.pad(thresholded, ((0, 0), (eps_t, eps_t)), mode='constant', constant_values=0)
    return padded

def tdscan(bin_flowers, time, tdscan_list, eps_t, min_pts):
    num_flowers = len(bin_flowers)
    tdscan_flowers = np.zeros((num_flowers, time), dtype=np.float32)
    # Precompute cumulative sum over time for each pixel
    cumsum_l1 = np.pad(np.cumsum(bin_flowers, axis=1), ((0, 0), (1, 0)))  # pad to handle start_t = 0 cleanly

    for i in range(num_flowers):
        neighbors = tdscan_list[i]
        if len(neighbors) == 0:
            continue
        # Compute rolling sum using cumsum
        start = np.arange(time) - eps_t
        end = np.arange(time) + eps_t + 1

        start = np.clip(start, 0, time)
        end = np.clip(end, 0, time)

        # Get total values in the window for each time frame using broadcasting
        total = cumsum_l1[neighbors][:, end] - cumsum_l1[neighbors][:, start]  # shape: (num_neighbors, time)
        total_sum = np.sum(total, axis=0)  # sum across neighbors
        tdscan_flowers[i] = (total_sum >= min_pts).astype(np.float32)
    return tdscan_flowers # shape: (n_flowers, time)

def apply_low_trigger(x, l1_list, l1_threshold, tdscan_list, eps_t, min_pts, trigger_type, l1_mask, max_pix, feb_indices, quant_step):
    """
    Apply a low level trigger to the raw waveform.

    This method applies a low level trigger to the raw waveform, the trigger options are a L1 digital sum
    per flower, superflower or FEB, and a L2 algorithm. like tdscan

    Parameters
    ----------
    x : 
        An array containing the complete raw waveform.
    trigger_type :
        String indicating which kind of trigger to apply, only l1, l1 + tdscan.
    l1_threshold : 
        Threshold in ADC counts for the digital sum.
    eps_t : 
        Set the the number of samples before and after to consider in the convolution. "
    min_pts : 
        Threshold in binary flower counts above which the convolution is going to keep the central flower. "

    Returns
    -------
    trigger : int
        Trigger binary mask.
    """
    n_pixels, time = x.shape
    peak_sample = np.argmax(x[max_pix])

    # Perform the first basic digital sum on the indicated region size.
    flower_sums = digital_sum(x, l1_list, l1_mask)
    bin_flowers = (flower_sums >= l1_threshold).astype(int)
    # Digital sum experiments
    if trigger_type == "rolling_l1":
        flower_sums = rolling_digital_sum(flower_sums, eps_t)
    elif trigger_type == "quantised_l1":
        flower_sums, ampmax = quantised_digital_sum(flower_sums, l1_threshold, bin_flowers, quant_step)
    elif trigger_type == "quantised_rolled_digital_sum":
        flower_sums, ampmax = quantised_rolled_digital_sum(flower_sums, l1_threshold, bin_flowers)
    elif trigger_type == 'quantised_per_feb':
        flower_sums, ampmax = quantised_per_feb_digital_sum(flower_sums, l1_threshold, quant_step)

    # Thresholding
    bin_flowers = (flower_sums > l1_threshold).astype(int)  # shape: (n_flowers, time)

    # L2 local trigger algorithms
    if trigger_type == "tdscan_perflower":
        bin_flowers = tdscan_per_flower(bin_flowers, eps_t, min_pts)
    elif trigger_type == "tdscan":
        bin_flowers = tdscan(bin_flowers, time, tdscan_list, eps_t, min_pts)

    if trigger_type in ['quantised_l1', 'quantised_rolled_digital_sum', 'quantised_per_feb']:
        if trigger_type in ['quantised_l1', 'quantised_rolled_digital_sum']:
            amps = flower_sums[:,:,1] 
            length = 128
        else:
            amps = flower_sums[:,:]
            length = 1024
        bin_flowers = amps >= min_pts
        feb_dist = np.bincount(amps.flatten(), minlength=length)
        feb_on = np.any(bin_flowers, axis=1)
        trigger = int(np.any(bin_flowers))
        trigger_per_sample = bin_flowers.any(axis=0).astype(int)
        return trigger, trigger_per_sample, bin_flowers, peak_sample, feb_on, ampmax, feb_dist
    else:
        # Get triggers per sample, feb and general
        rep = 49 if l1_list.shape[0]==163 else 7
        bin_pixels = np.repeat(bin_flowers, rep, axis=0)
        pixel_on = np.any(bin_pixels, axis=1) #7987
        feb_on = np.any(pixel_on[feb_indices], axis=1)
        trigger = int(np.any(bin_flowers))
        trigger_per_sample = bin_flowers.any(axis=0).astype(int)
        return trigger, trigger_per_sample, bin_flowers, peak_sample, feb_on

def get_table(input_file, tel_ids=[1, 2, 3, 4]):
    """
    Crea una tabla con todos los eventos para los telescopios indicados.
    Verifica que la tabla de r0 tenga los mismos eventos que la de simulación.
    """
    logger = logging.getLogger()
    
    tel_tables = []

    energy_path = "/simulation/event/subarray/shower"
    logger.info(f"Reading shower data from {energy_path}")
    try:
        energy_tab = read_table(input_file, energy_path)
        energy_tab = energy_tab['obs_id', 'event_id', 'true_energy']
    except Exception as e:
        logger.warning(f"  -> Failed to read shower table")

    for tel_id in tel_ids:
        wf_path = f"/r0/event/telescope/tel_{tel_id:03d}"
        sim_path = f"/simulation/event/telescope/images/tel_{tel_id:03d}"

        # logger.info(f"Reading R0 data for tel_{tel_id:03d} from {wf_path}")
        # try:
        #     r0 = read_table(input_file, wf_path)
        # except Exception as e:
        #     logger.warning(f"  -> Failed to read R0 table for tel_{tel_id:03d}: {e}")
        #     continue
        logger.info(f"Reading sim data for tel_{tel_id:03d} from {sim_path}")
        try:
            sim = read_table(input_file, sim_path)
        except Exception as e:
            logger.warning(f"  -> Failed to read simulation table for tel_{tel_id:03d}: {e}")
            continue

        # if len(r0) == 0:
        #     logger.warning(f"  -> R0 table for tel_{tel_id:03d} is empty, skipping")
        #     continue

        # logger.info(f"  -> {len(r0)} rows in R0 table for tel_{tel_id:03d}")
        logger.info(f"  -> {len(sim)} rows in sim table for tel_{tel_id:03d}")

        # if len(r0) != len(sim):
        #     raise ValueError(f"Simulation table and r0 table have not the same length for tel_{tel_id:03d}")
        # if len(np.unique(r0["event_id"])) != len(r0):
        #     raise ValueError(f"R0 table has duplicated events for tel_{tel_id:03d}")
        if len(np.unique(sim["event_id"])) != len(sim):
            raise ValueError(f"Sim table has duplicated events for tel_{tel_id:03d}")
        # if not np.array_equal(np.sort(r0["event_id"]), np.sort(sim["event_id"])):
        #     raise ValueError(f"Mismatch in event_ids between R0 and Sim tables for tel_{tel_id:03d}")

        # Añadir columna de índice para cada fila
        sim = sim["obs_id", "event_id", "tel_id", "true_image_sum"]
        sim.add_column(np.arange(len(sim)), name="table_index", index=0)

        tel_tables.append(sim)
        # r0 = r0["obs_id", "event_id", "tel_id"]
        # r0.add_column(np.arange(len(r0)), name="table_index", index=0)

        # tel_tables.append(r0)

    if len(tel_tables) == 0:
        logger.error("No valid telescope R0 tables were loaded!")
        return None

    logger.info("Stacking all telescope R0 tables")
    full_data = vstack(tel_tables)
    logger.info("Merging all telescope tables with energy information.")
    full_data = join(
        left=full_data,
        right=energy_tab,
        keys=["obs_id", "event_id"],
    )
    full_data.sort(['obs_id', 'event_id', 'tel_id'])
    logger.info(f"Full table has {len(full_data)} rows and columns: {full_data.colnames}")

    return full_data



if __name__ == "__main__":
    main()

