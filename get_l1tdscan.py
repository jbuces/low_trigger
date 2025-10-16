import numpy as np
import os
import sys
import argparse
from eventio import SimTelFile
import tables
from ctapipe.io import read_table
from astropy.table import Table, join, vstack
import logging
from numpy.lib.stride_tricks import sliding_window_view

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger()


help_message = """
This script reads a simtel file applies the low level trigger (digital sum + tdscan) and gets the stats.
Example usage:
nohup python3 get_l1tdscan_stats.py -i /lustre/ific.uv.es/ml/ucm147/datasets/TriggerlessData/nsb/NSB268MHZ/corsika_run0.r0.h5 -o /lhome/ext/ucm147/ucm1473/lowtrigger_stats/nsb/corsika_run0_2161cut_tdscan7.h5 -nf /lhome/ext/ucm147/ucm1473/dl1-data-handler/dl1_data_handler/ressources/newcam_sipm_patches.h5 -pts 7 -l1 2161 -epsl1 1 -trig tdscan -epst 1 > /lhome/ext/ucm147/ucm1473/lowtrigger_stats/nsb/corsika_run0_2161cut_tdscan7.log 2>&1 &
"""

def parse_args():
    parser = argparse.ArgumentParser(description=help_message)
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-add', '--add_mask', help='Boolean to add the mask to the table', required=True, default=True)
    parser.add_argument('-nf', '--neighbors_file', help='Neighbors file for the l1 and tdscan', required=True)
    parser.add_argument('-l1', '--l1_threshold', help='Threshold for the l1 trigger', required=True, type=int)
    parser.add_argument('-epsl1', '--epsl1', help='Use flower (1) or superflower (2) for digital sum, (3) for non overlapping superflowers (modules)', required=True, type=int)
    parser.add_argument('-trig', '--trigger_type', help='l1 for digital sum only, tdscan for tdscan algo on top', required=True, type=str, default="tdscan")
    parser.add_argument('-epst', '--epst', help='Frames before and after for tdscan', required=True, type=int)
    parser.add_argument('-pts', '--min_pts', help='Min pts for tdscan flower convolution', required=True, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    simtel_file = args.input
    out_table = args.output
    epsl1 = args.epsl1
    l1_threshold = args.l1_threshold
    neighbors_file = args.neighbors_file
    eps_t = args.epst
    min_pts = args.min_pts
    trigger_type = args.trigger_type
    add_mask = args.add_mask

    output_dir = os.path.dirname(out_table)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    l1_array, tdscan_list = get_neighbors_info(neighbors_file, epsl1)
    if epsl1==2 or epsl1==3:
        l1_mask = (l1_array >= 0).astype(int)
    else:
        l1_mask = None
    trigtable = get_table(simtel_file)
    n_events = len(trigtable)
    l1_trigger, trig_per_sample = [], []
    logger.info("Processing the data...")

    packed_masks = None
    n_pixels = None
    n_times = None
    bits = None
    packed_len = None
    bitorder = "big"

    with tables.open_file(simtel_file, 'r') as h5file:
        for i, row in enumerate(trigtable):
            tel_id = row["tel_id"]
            table_idx = row["table_index"]
            tel_path = f'/r0/event/telescope/tel_{tel_id:03d}'
            wf = h5file.get_node(tel_path)[table_idx]['waveform']

            trig, trig_p_sample, bin_flowers = apply_low_trigger(wf, l1_array, l1_threshold, tdscan_list, eps_t, min_pts, trigger_type, l1_mask)
            l1_trigger.append(trig)
            trig_per_sample.append(trig_p_sample)

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

    # l1_list = (
    #     fl_neighbor_l1_list 
    #     if epsl1 == 1 
    #     else supfl_neighbor_array_l1
    # )
    if epsl1==1:
        l1_list=fl_neighbor_array_l1
    elif epsl1==2:
        l1_list=supfl_neighbor_array_l1
    else:
        l1_list=feb_indices
        neighbor_tdscan_eps1_list= feb_neighbors

    return l1_list, neighbor_tdscan_eps1_list

def apply_low_trigger(x, l1_list, l1_threshold, tdscan_list, eps_t, min_pts, trigger_type, l1_mask):
    """
    Apply a low level trigger to the raw waveform.

    This method applies a low level trigger to the raw waveform, the trigger options are a digital sum
    per flower or superflower named ``l1``, or a ``tdscan`` algorithm.

    Parameters
    ----------
    x : r0_event : np.ndarray
        An array containing the complete raw waveform.
    trigger_type : string
        String indicating which kind of trigger to apply, l1 or l1+tdscan.
    l1_threshold : 
        Threshold in ADC counts for the digital sum.
    eps_t : 
        Set the the number of samples before and after to consider in the convolution. "
    min_pts : 
        Threshold in binary flower counts above which the convolution is going to keep the central flower. "

    Returns
    -------
    trigger : int
        Trigger binary.
    """
    x=x[0,:,:]
    n_pixels, length = x.shape
    sequence_position = length // 2 - 1            
    init = int(1 + sequence_position - 30 / 2)
    stop =  int(1 + sequence_position + 30 / 2)
    x = x[:, int(init) : int(stop)]
    time = x.shape[1]
    # Assign the l1 neighbor list, summing on flowers or superflowers for each flower.
    if l1_mask is not None:
        vals = x[l1_list] 
        flower_sums = (vals * l1_mask[..., None]).sum(axis=1)
    else:
        flower_sums = x[np.array(l1_list)].sum(axis=1)

    # Thresholding (broadcast back to each pixel in group)
    bin_flowers = (flower_sums > l1_threshold).astype(int)  # shape: (n_flowers, time)
    if trigger_type == "tdscan_perflower":
        windows = sliding_window_view(bin_flowers, window_shape=min_pts, axis=1)
        valid = np.all(windows == 1, axis=2)
        triggered = np.zeros_like(bin_flowers)
        for offset in range(min_pts):
            triggered[:, offset:offset + valid.shape[1]] |= valid
        bin_flowers=triggered
    # TDSCAN mask cleaning:
    if trigger_type == "tdscan":
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
        bin_flowers = tdscan_flowers # shape: (n_flowers, time)

    trigger = int(np.any(bin_flowers))
    trigger_per_sample = bin_flowers.any(axis=0).astype(int)

    return trigger, trigger_per_sample, bin_flowers

def get_table(input_file, tel_ids=[1, 2, 3, 4]):
    """
    Crea una tabla con todos los eventos para los telescopios indicados.
    Verifica que la tabla de r0 tenga los mismos eventos que la de simulación.
    """
    tel_tables = []

    energy_path = "/simulation/event/subarray/shower"
    logger.info(f"Reading shower data from {energy_path}")
    try:
        energy_tab = read_table(input_file, energy_path)
        energy_tab = energy_tab['obs_id', 'event_id', 'true_energy']
    except Exception as e:
        logger.warning(f"  -> Failed to read simulation table for tel_{tel_id:03d}: {e}")

    for tel_id in tel_ids:
        wf_path = f"/r0/event/telescope/tel_{tel_id:03d}"
        sim_path = f"/simulation/event/telescope/images/tel_{tel_id:03d}"

        logger.info(f"Reading R0 data for tel_{tel_id:03d} from {wf_path}")
        try:
            r0 = read_table(input_file, wf_path)
        except Exception as e:
            logger.warning(f"  -> Failed to read R0 table for tel_{tel_id:03d}: {e}")
            continue
        logger.info(f"Reading sim data for tel_{tel_id:03d} from {sim_path}")
        try:
            sim = read_table(input_file, sim_path)
        except Exception as e:
            logger.warning(f"  -> Failed to read simulation table for tel_{tel_id:03d}: {e}")
            continue

        if len(r0) == 0:
            logger.warning(f"  -> R0 table for tel_{tel_id:03d} is empty, skipping")
            continue

        logger.info(f"  -> {len(r0)} rows in R0 table for tel_{tel_id:03d}")
        logger.info(f"  -> {len(sim)} rows in sim table for tel_{tel_id:03d}")

        if len(r0) != len(sim):
            raise ValueError(f"Simulation table and r0 table have not the same length for tel_{tel_id:03d}")
        if len(np.unique(r0["event_id"])) != len(r0):
            raise ValueError(f"R0 table has duplicated events for tel_{tel_id:03d}")
        if len(np.unique(sim["event_id"])) != len(sim):
            raise ValueError(f"Sim table has duplicated events for tel_{tel_id:03d}")
        if not np.array_equal(np.sort(r0["event_id"]), np.sort(sim["event_id"])):
            raise ValueError(f"Mismatch in event_ids between R0 and Sim tables for tel_{tel_id:03d}")

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

