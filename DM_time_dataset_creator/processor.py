import os
import glob
import numpy as np
import pandas as pd
import json
from decimal import Decimal
from tqdm import tqdm
import your
from numba import jit
import gc
import math

class DMTimeDataSetCreator:
    """
    A class for creating a DM-Time dataset from transient candidates.

    Attributes:
        config_path (str): Path to the configuration JSON file.
        filterbank_file (your.Your): Filterbank file object from the `your` library.
        transient_x_path (str): Path to the directory containing time-series for different DM trials.
        transient_x_cands_path (str): Path to the file containing list of candidates.
        dm_ranges (dict): Left and right edges of DM range.
        ntsamples (int): Number of time samples for each candidate.
        output_dir (str): Directory to store the output dataset and labels.
        name_of_set (str): Base name of the dataset based on the filterbank file name.
    """
    def __init__(self, config_path):
        """
        Initialize the DMTimeDataSetCreator with the given configuration file.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        # Load configuration
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        # Extract paths and parameters from config
        self.filterbank_file = your.Your(self.config["filterbank_path"])
        self.transient_x_path = self.config["transientx_time_series_path"]
        self.transient_x_cands_path = self.config["transientx_candidates_path"]
        self.dm_ranges = self.config["dm_ranges"]
        self.ntsamples = self.config["ntsamples"]
        self.output_dir = os.path.join(os.getcwd(), 'outputs')
        self.name_of_set = self.filterbank_file.your_header.basename
        shard_size_gb = self.config.get("max_shard_size_gb", self.config.get("shard_size_gb", 20))
        shard_size_gb = float(shard_size_gb)
        if shard_size_gb <= 0:
            raise ValueError("max_shard_size_gb must be a positive number")
        self.max_shard_size_bytes = int(shard_size_gb * (1024 ** 3))
        self.dm_time_shard_dir = os.path.join(self.output_dir, 'dm_time_shards')
        self.freq_time_shard_dir = os.path.join(self.output_dir, 'dedispersed_freq_time_shards')
        

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dm_time_shard_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.dm_time_shard_dir, "test"), exist_ok=True)
        os.makedirs(os.path.join(self.freq_time_shard_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.freq_time_shard_dir, "test"), exist_ok=True)

        # Prepare file list and DMs
        self.file_list, self.dm_list = self._prepare_file_list_and_dm()

        # Load DM-time image
        self.dm_time_image = self._create_dm_time_image()

    @staticmethod
    def normalize_image_to_255(image):
        """
        Normalize a 2D image to the 0-255 range.

        Args:
            image (numpy.ndarray): Input image to normalize.

        Returns:
            numpy.ndarray: Normalized image scaled to 0-255 as uint8.
        """
        image = image.astype(np.float32)
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return (normalized_image * 255).astype(np.uint8)

    def _prepare_file_list_and_dm(self):
        """
        Prepare a sorted list of DM data files and their corresponding DM values.

        Returns:
            tuple: A tuple containing a list of file paths and a list of DM values.
        """
        flist = sorted(
            glob.glob(os.path.join(self.transient_x_path, '*.dat')),
            key=lambda x: float(os.path.basename(x).split('DM')[1].split('.dat')[0])
        )
        dms = [float(os.path.basename(i).split('DM')[1].split('.dat')[0]) for i in flist]
        return flist, dms

    def _create_dm_time_image(self):
        """
        Create the DM-Time image from the list of .dat files.

        Returns:
            numpy.ndarray: 2D array where each row corresponds to time series from a .dat file for a specific DM value.
        """
        array_size = np.fromfile(self.file_list[0], dtype=np.float32).size
        dm_time_image = np.empty((len(self.file_list), array_size))

        for idx, file in tqdm(enumerate(self.file_list), total=len(self.file_list), desc='Processing files'):
            dm_time_image[idx] = np.fromfile(file, dtype=np.float32)

        return dm_time_image

    def _save_dm_time_shards(self, data, name):
        """
        Persist the DM-Time dataset as ~20 GB shards under the dedicated output folder.

        Args:
            data (np.ndarray): Array shaped (N/nt_samples, nt_samples, n_dm, 256) with dtype uint8.
        """
        if data.size == 0:
            return

        if data.ndim != 3:
            raise ValueError("Expected DM-Time data with shape (N*nt_samples, n_dm, 256)")

        bytes_per_sample = data.shape[1] * data.shape[2] * data.dtype.itemsize
        if bytes_per_sample == 0:
            raise ValueError("Unable to compute shard size for empty samples")

        samples_per_shard = max(1, self.max_shard_size_bytes // bytes_per_sample)

        total_samples = data.shape[0]
        shard_idx = 0
        shard_infos = []

        for start in range(0, total_samples, samples_per_shard):
            end = min(start + samples_per_shard, total_samples)
            shard_path = os.path.join(
                self.dm_time_shard_dir, name,
                f"{self.name_of_set}_DM_time_dataset_realbased_shard_{shard_idx:04d}.npy"
            )
            np.save(shard_path, data[start:end], allow_pickle=False)
            shard_infos.append(
                {
                    "index": shard_idx,
                    "path": os.path.basename(shard_path),
                    "start_sample": int(start),
                    "end_sample": int(end),
                    "rows": int(end - start),
                }
            )
            shard_idx += 1

        manifest = {
            "dataset": "dm_time",
            "name": self.name_of_set,
            "dtype": str(data.dtype),
            "sample_shape": (int(data.shape[1]), int(data.shape[2])),
            "total_samples": int(total_samples),
            "max_shard_size_bytes": int(self.max_shard_size_bytes),
            "shards": shard_infos,
        }

        manifest_path = os.path.join(
            self.dm_time_shard_dir, name,
            f"{self.name_of_set}_DM_time_dataset_manifest.json"
        )
        with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

        print(f"Saved {shard_idx} DM-time shard(s) to {self.dm_time_shard_dir}")
        print(f"DM-time shard manifest written to {manifest_path}")
        
    @staticmethod
    def _shuffle_in_unison(arrays, seed=None):
        if not arrays:
            return
        length = arrays[0].shape[0]
        if length < 2:
            return
        for arr in arrays[1:]:
            if arr.shape[0] != length:
                raise ValueError("All arrays must share the same length to shuffle together.")
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(length)
        for arr in arrays:
            np.take(arr, permutation, axis=0, out=arr)
            
            
    def _get_position_in_filfile(self, mjd_pulse):
        """
        Calculate the position of a pulse in the filterbank file based on its MJD.

        Args:
            mjd_pulse (float): MJD of the pulse.

        Returns:
            int: Position in the filterbank file corresponding to the given MJD.
        """
        mjd_pulse = Decimal(mjd_pulse)
        mjd_start = Decimal(self.filterbank_file.your_header.tstart)
        delta_t_mjd = mjd_pulse - mjd_start
        delta_t_seconds = delta_t_mjd * 86400
        location_in_the_file = delta_t_seconds / Decimal(self.filterbank_file.your_header.tsamp)
        return int(round(location_in_the_file, 0))

    def _sample_index_to_time_bounds(self, start_index, width):
        """
        Convert a starting sample index and slice width to time bounds in seconds since filterbank start.

        Args:
            start_index (int): Starting sample index of the slice.
            width (int): Number of samples in the slice.

        Returns:
            tuple: (t_min_seconds, t_max_seconds)
        """
        tsamp = float(self.filterbank_file.your_header.tsamp)
        t_min = max(start_index, 0) * tsamp
        t_max = (max(start_index, 0) + width) * tsamp
        return t_min, t_max
    

    def _process_candidates(self, candidates, label, exclude_positions=None, target_count=None):
        """
        Process candidates to extract data or generate random segments for the 'rest' category.

        Args:
            candidates (pd.DataFrame): DataFrame of candidates to process.
            label (str): Category label ('pulses', 'zero DM events', or 'rest of events').
            exclude_positions (list, optional): List of position ranges to exclude for 'rest of events'.
            target_count (int, optional): Target number of samples for 'rest of events'.

        Returns:
            tuple: Dataset array and metadata array shaped (N, 3) with columns ('snr', 't_min', 't_max', 'DM').
        """
        if label == 'rest of events':
            # Ensure target count is provided
            if target_count is None:
                raise ValueError("Target count must be specified for 'rest of events'")
    
            # Initialize dataset with the target count
            dataset = np.empty([math.floor(target_count/self.ntsamples),self.ntsamples, len(self.dm_list), 256], dtype=np.uint8)
            metadata = np.empty((math.floor(target_count/self.ntsamples),self.ntsamples, 4), dtype=np.float64) # SNR, t_min, t_max, DM per sample
            target_count = math.floor(target_count/self.ntsamples) * self.ntsamples
            
            global_index : int = 0 # make sure division is int
            self.ntsamples = int(self.ntsamples)
            
            
            # Use tqdm to display progress bar
            with tqdm(total=target_count, desc='Processing rest of events') as pbar:
                while global_index < target_count:
                    # Generate a random position
                    position = np.random.randint(0, self.dm_time_image.shape[1] - 256)
                    
                    # Check if position overlaps with pulses or artefacts
                    if not self._is_in_pulse_or_bbrfi(position, exclude_positions): #TODO try to add positions after processing to exclude_positions, so that noise segments are unique
                        slice_start = position
                        slice_end = slice_start + 256
                        t_min, t_max = self._sample_index_to_time_bounds(slice_start, slice_end - slice_start)
                        dataset[global_index // self.ntsamples, global_index % self.ntsamples] = self.normalize_image_to_255(
                            self.dm_time_image[:, slice_start:slice_end][::-1]
                        )
                        #scan_snr = self._get_scansnr(self.dm_time_image[:, slice_start:slice_end][::-1])
                        dm = self._getDM(position)
                        metadata[global_index // self.ntsamples, global_index % self.ntsamples] = (np.nan, t_min, t_max, dm)
                        global_index += 1
                        pbar.update(1)  # Update progress bar
                
            return dataset, metadata
        else:
            # Process pulses or zero DM events
            dataset = np.empty([candidates.shape[0], self.ntsamples, len(self.dm_list), 256], dtype=np.uint8)
            metadata = np.empty((candidates.shape[0], self.ntsamples, 4), dtype=np.float64)
            global_index = 0
    
            for idx, row in tqdm(candidates.iterrows(), total=candidates.shape[0], desc=f'Processing {label}'):
                position = int(self._get_position_in_filfile(row['mjd']))
                for i in range(self.ntsamples):
                    slice_start = position - i
                    slice_start = max(slice_start, 0)
                    slice_end = slice_start + 256
                    if slice_end > self.dm_time_image.shape[1]:
                        slice_end = self.dm_time_image.shape[1]
                        slice_start = slice_end - 256
                    t_min, t_max = self._sample_index_to_time_bounds(slice_start, slice_end - slice_start)
                    dataset[global_index // self.ntsamples, global_index % self.ntsamples] = self.normalize_image_to_255(
                        self.dm_time_image[:, slice_start:slice_end][::-1]
                    )
                    metadata[global_index // self.ntsamples, global_index % self.ntsamples] = (float(row['snr']), t_min, t_max, float(row['dm']))
                    global_index += 1
    
            return dataset, metadata



    def _is_in_pulse_or_bbrfi(self, position, list_of_position):
        """
        Check if a position overlaps with any pulse or BBRFI positions.

        Args:
            position (int): Position to check.
            list_of_position (list): List of (start, end) position ranges.

        Returns:
            bool: True if position overlaps, False otherwise.
        """
        for start, end in list_of_position:
            if start <= position <= end:
                return True
        return False
    
    def _generate_exclude_positions(self, pulse_candidates, zero_dm_events):
        """
        Generate a list of positions to exclude based on pulse and BBRFI.

        Args:
            pulse_candidates (pd.DataFrame): DataFrame of pulse candidates.
            zero_dm_events (pd.DataFrame): DataFrame of zero DM events.

        Returns:
            list: List of (start, end) position ranges to exclude.
        """
        exclude_positions = []
    
        # Add pulse positions
        for _, row in pulse_candidates.iterrows():
            position = int(self._get_position_in_filfile(row['mjd']))
            exclude_positions.append((position - 256, position))
    
        # Add zero DM event positions
        for _, row in zero_dm_events.iterrows():
            position = int(self._get_position_in_filfile(row['mjd']))
            exclude_positions.append((position - 256, position))
    
        return exclude_positions


    def process(self, train_ratio, shuffle=True, shuffle_seed=None):
        """
        Main function to process candidates and create the DM-Time dataset.

        Args:
            shuffle (bool, optional): Whether to shuffle before sharding. Defaults to True.
            shuffle_seed (int, optional): Seed for deterministic shuffling.

        Steps:
            1. Load and categorize candidates.
            2. Process each category ('pulses', 'zero DM events', 'rest of events').
            3. Combine datasets, shuffle, and persist DM-time shards plus metadata/labels arrays.

        Outputs:
            - DM-time dataset persisted as shards (~20 GB) under `dm_time_shards/`.
            - Corresponding labels as a .npy file.
            - Metadata (snr, t_min, t_max, DM) as a .npy file.
        """
        if train_ratio < 0 or train_ratio >1:
            return ValueError("train_ratio is not between 0 and 1")
        # Load candidates
        column_names = ['beam_name', 'nn', 'mjd', 'dm', 'width', 'snr', 'fh', 'fl', 'image_name', 'x', 'name_file']
        candidats = pd.read_csv(self.transient_x_cands_path, sep='\t', names=column_names, dtype=str)
        candidats['dm'] = candidats['dm'].astype(float)
        candidats['snr'] = candidats['snr'].astype(float)
        candidats.sort_values(by='snr', inplace=True)
    
        # Categorize candidates
        pulses_range = self.dm_ranges["pulses"]
        zero_dm_events = candidats[candidats['dm'] == 0]
        pulse_candidates = candidats[(pulses_range[0] <= candidats['dm']) & (candidats['dm'] <= pulses_range[1])]
        rest = candidats[(candidats['dm'] != 0) & ((pulses_range[0] > candidats['dm']) | (candidats['dm'] > pulses_range[1]))]
    
        # Generate positions to exclude
        exclude_positions = self._generate_exclude_positions(pulse_candidates, zero_dm_events)
    
        # Process each category
        dataset_with_pulses, metadata_pulses = self._process_candidates(pulse_candidates, 'pulses') #(N_pulses, nt_samples, len_DM_list = 256, 256)
        dataset_with_bbrfi, metadata_bbrfi = self._process_candidates(zero_dm_events, 'zero DM events') #(N_bbrfi, nt_samples, len_DM_list = 256, 256)
        
        target_count = dataset_with_pulses.shape[0] * dataset_with_pulses.shape[1] + dataset_with_bbrfi.shape[0] * dataset_with_bbrfi.shape[1]

        # Generate rest of events
        dataset_with_rest, metadata_rest = self._process_candidates( #(N_restofevents/nt_samples,nt_samples, len_DM_list = 256, 256) # N_restofevents = N_pulses + N_bbrfi >> 2 (ca. 2300 bei snr_thr = 3)
            rest, 
            'rest of events', 
            exclude_positions=exclude_positions, 
            target_count=target_count
)
        print("combine data")
        # Combine datasets and create labels
        data = np.concatenate((dataset_with_pulses, dataset_with_bbrfi, dataset_with_rest), axis=0)
        print("combine metadata")
        metadata = np.concatenate((metadata_pulses, metadata_bbrfi, metadata_rest), axis=0)
        print("combine labels")
        labels = np.array(
            [['Pulse'] * self.ntsamples] * len(dataset_with_pulses) +
            [['Artefact']* self.ntsamples] * (len(dataset_with_bbrfi) + len(dataset_with_rest))
        )
        print(f"labels shape: {labels.shape}")
        if shuffle:
            print("Shuffling data, metadata, and labels before sharding ...")
            self._shuffle_in_unison([data, metadata, labels], seed=shuffle_seed)
            print("Shuffling finished")
        else:
            print("Warning: dataset is not shuffled before sharding. Pass shuffle=True to randomize order of pulses.")
            
        #seperating into training and test,
        # so that we can turn back data shape from (N, ntsamples, dm_length, 256) back to (N * ntsamples, dm_length, 256), which we need for sharding 
        
        data_len = len(data)
        data_train = data[:math.floor(train_ratio * data_len)].reshape(-1, len(self.dm_list), 256)
        data_test = data[math.floor(train_ratio * data_len):].reshape(-1, len(self.dm_list), 256)
        metadata_train = metadata[:math.floor(train_ratio * data_len)].reshape(-1, 4)
        metadata_test = metadata[math.floor(train_ratio * data_len):].reshape(-1, 4)
        labels_train = labels[:math.floor(train_ratio * data_len)].reshape(-1)
        labels_test = labels[math.floor(train_ratio * data_len):].reshape(-1)
        
        
        if shuffle:
            print("shuffling inside train and test before sharding, so that data more evenly distributed")
            self._shuffle_in_unison([data_train, metadata_train, labels_train], seed=shuffle_seed)
            self._shuffle_in_unison([data_test, metadata_test, labels_test], seed=shuffle_seed)
        
        print("data_train.shape: ", data_train.shape)
        print("data_test.shape: ", data_test.shape)
        print("metadata_train.shape: ", metadata_train.shape)
        print("metadata_test.shape: ", metadata_test.shape)
        print("labels_train.shape: ", labels_train.shape)
        print("labels_test.shape: ", labels_test.shape)
        
        
        # Save DM-time dataset shards
        self._save_dm_time_shards(data_train, "train")
        self._save_dm_time_shards(data_test, "test")
        del data
        del data_train
        del data_test
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased_labels_train.npy'), labels_train)
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased_metadata_train.npy'), metadata_train)
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased_labels_test.npy'), labels_test)
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased_metadata_test.npy'), metadata_test)

    def report_snr_statistics(self, metadata_path=None, labels_path=None, snr_round=2, ratio = None, test = None):
        """
        Load saved labels and metadata arrays and print SNR-related statistics.

        Args:
            metadata_path (str, optional): Path to metadata .npy file. Defaults to creator output.
            labels_path (str, optional): Path to labels .npy file. Defaults to creator output.
            snr_round (int, optional): Decimal places when grouping pulsar SNR counts. Defaults to 2.
        """
        if metadata_path is None:
            metadata_path = os.path.join(
                self.output_dir,
                f'{self.name_of_set}_DM_time_dataset_realbased_metadata.npy'
            )
        if labels_path is None:
            labels_path = os.path.join(
                self.output_dir,
                f'{self.name_of_set}_DM_time_dataset_realbased_labels.npy'
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        else:
            print(f"using {metadata_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
        else:
            print(f"using {labels_path}")

        metadata = np.load(metadata_path)
        labels = np.load(labels_path)
    
        
        if ratio is not None and test is not None and ratio > 0.0 and ratio < 1.0:
            length = len(metadata)
            if test:
                metadata = metadata[math.floor(length*ratio):]
                labels = labels[math.floor(length*ratio):]
            else:
                metadata = metadata[:math.floor(length*ratio)]
                labels = labels[:math.floor(length*ratio)]
        
        print(metadata.shape)

        if metadata.ndim != 2 or metadata.shape[1] < 4:
            raise ValueError(
                "Expected metadata with shape (N, 4) where columns are (snr, t_min, t_max, DM)."
            )
        if labels.shape[0] != metadata.shape[0]:
            raise ValueError(
                f"Label count ({labels.shape[0]}) does not match metadata count ({metadata.shape[0]})."
            )

        snr_values = metadata[:, 0]
        total_samples = labels.shape[0]
        finite_mask = np.isfinite(snr_values)
        finite_snr = snr_values[finite_mask]
        pulse_mask = labels == 'Pulse'
        pulse_count = int(np.sum(pulse_mask))

        print("=== Dataset Overview ===")
        print(f"Total samples: {total_samples}")
        print(f"Samples with finite SNR: {finite_snr.size} ({finite_snr.size / total_samples:.2%})")
        print(f"Samples with NaN/inf SNR: {total_samples - finite_snr.size}")
        if total_samples:
            share = pulse_count / total_samples
            print(f"Pulsar samples: {pulse_count} ({share:.2%} of all samples)")
        if finite_snr.size:
            finite_pulse_count = int(np.sum(pulse_mask & finite_mask))
            share = finite_pulse_count / finite_snr.size
            print(
                f"Pulsar entries with finite SNR: {finite_pulse_count} "
                f"({share:.2%} of finite-SNR samples)"
            )
        else:
            finite_pulse_count = 0

        if finite_snr.size:
            print("\n=== SNR Summary (finite values) ===")
            print(f"Min SNR: {np.min(finite_snr):.2f}")
            print(f"Max SNR: {np.max(finite_snr):.2f}")
            print(f"Mean SNR: {np.mean(finite_snr):.2f}")
            print(f"Median SNR: {np.median(finite_snr):.2f}")
            q10, q90 = np.percentile(finite_snr, [10, 90])
            print(f"10th percentile: {q10:.2f}")
            print(f"90th percentile: {q90:.2f}")

            print("\n=== SNR Spectrum ===")
            negative_mask = finite_snr < 0
            negative_count = int(np.sum(negative_mask))
            if negative_count:
                negative_share = negative_count / finite_snr.size
                print(f"SNR < 0: count={negative_count}, share={negative_share:.2%}")

            non_negative_snr = finite_snr[~negative_mask]
            if non_negative_snr.size:
                bucket_ids = np.floor(non_negative_snr).astype(int)
                start_bucket = int(bucket_ids.min())
                start_bucket = max(0, start_bucket)
                end_bucket = int(bucket_ids.max())
                for snr_int in range(start_bucket, end_bucket + 1):
                    count = int(np.sum(bucket_ids == snr_int))
                    share = count / finite_snr.size if finite_snr.size else 0
                    print(f"SNR {snr_int}: count={count}, share={share:.2%}")
            else:
                print("No non-negative SNR values available.")
        else:
            print("\nNo finite SNR values available for statistics.")

        pulse_snr = snr_values[pulse_mask & finite_mask]

        print("\n=== Pulsar Counts by SNR ===")
        if pulse_snr.size:
            natural_bins = np.floor(pulse_snr)
            negative_mask = natural_bins < 0
            total_pulsars = pulse_snr.size

            if np.any(negative_mask):
                neg_count = int(np.sum(negative_mask))
                neg_share = neg_count / total_pulsars if total_pulsars else 0
                print(f"SNR < 0: {neg_count} pulsars ({neg_share:.2%})")

            non_negative_bins = natural_bins[~negative_mask]
            if non_negative_bins.size:
                min_bucket = int(non_negative_bins.min())
                min_bucket = max(0, min_bucket)
                max_bucket = int(non_negative_bins.max())
                for snr_int in range(min_bucket, max_bucket + 1):
                    count = int(np.sum(non_negative_bins == snr_int))
                    share = count / total_pulsars if total_pulsars else 0
                    print(f"SNR {snr_int}: {count} pulsars ({share:.2%})")
            else:
                print("No pulsar entries with non-negative SNR.")
        else:
            print("No pulsar entries with finite SNR found.")
            
                
    def get_dedispersed_freq_time(self, name, metadata_path=None, patch_width=256, use_memmap=True):
        """
        Create dedispersed freq-time data and save it in ~20 GB shards.

        Each shard is stored as a separate `.npy` file under the
        `dedispersed_freq_time_shards/` directory with shape
        `(num_samples_in_shard, nchans, patch_width)` and dtype `uint8`.
        The `use_memmap` flag is retained for compatibility but shards are always
        written through memory-mapped arrays to limit RAM usage.
        """
        
        #brauchen das nicht; lieber rausschmeißen bevor es Speicher zu müllt
        for attr in ("file_list", "dm_list", "dm_time_image"):
            if hasattr(self, attr):
                delattr(self, attr)

        if metadata_path is None:
            metadata_path = os.path.join(
                self.output_dir,
                f'{self.name_of_set}_DM_time_dataset_realbased_metadata_{name}.npy'
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        metadata = np.load(metadata_path)

        if metadata.ndim != 2 or metadata.shape[1] < 4:
            raise ValueError(
                "Expected metadata with shape (N, 4) where columns are (snr, t_min, t_max, DM)."
            )

        hdr = self.filterbank_file.your_header
        tsamp = float(hdr.tsamp)

        # for nchans
        test_block = self.filterbank_file.get_data(0, 1)  # (1, nchans)
        nchans = test_block.shape[1]
        
        total_samples = int(getattr(hdr, "nspectra", 0))

        # for dedispersion formula
        fch1 = float(hdr.fch1)   # MHz
        foff = float(hdr.foff)   # MHz/Kanal
        freqs = fch1 + np.arange(nchans) * foff  # MHz
        f_ref = np.max(freqs)
        k_dm = 4.148808e3  # s * MHz^2 * pc^-1 cm^3
        geom_term = (freqs**-2 - f_ref**-2)      # nur Frequenzabhängigkeit

        n_samples = metadata.shape[0]

        if n_samples == 0:
            print("No metadata entries found; skipping dedispersed freq-time export.")
            return

        bytes_per_sample = nchans * patch_width * np.dtype(np.uint8).itemsize
        samples_per_shard = max(1, self.max_shard_size_bytes // bytes_per_sample)

        shard_idx = 0
        processed = 0
        shard_infos = []

        with tqdm(total=n_samples, desc="Creating dedispersed freq-time patches") as pbar:
            while processed < n_samples:
                shard_start = processed
                shard_end = min(shard_start + samples_per_shard, n_samples)
                shard_size = shard_end - shard_start

                shard_path = os.path.join(
                    self.freq_time_shard_dir,name,
                    f"{self.name_of_set}_dedispersed_freq_time_shard_{shard_idx:04d}.npy"
                )
                shard_memmap = np.lib.format.open_memmap(
                    shard_path,
                    mode='w+',
                    dtype=np.uint8,
                    shape=(shard_size, nchans, patch_width)
                )

                for local_idx, idx in enumerate(range(shard_start, shard_end)):
                    snr, t_min, t_max, dm = metadata[idx]

                    # müssen Zeitfenster padden und anpassen, weil wir Punkte aus der "Zukunft" um negative Zeit verschieben
                    t_center = 0.5 * (t_min + t_max)
                    n_center = int(round(t_center / tsamp))

                    # Startindex von dedispersed Fenster
                    start = n_center - patch_width // 2
                    if start < 0:
                        start = 0
                    if start + patch_width > total_samples:
                        start = max(0, total_samples - patch_width)

                    if not np.isfinite(dm) or dm == 0.0:
                        shard_memmap[local_idx] = self.filterbank_file.get_data(start, patch_width).astype(np.uint8).T
                        if local_idx % 256 == 0:
                            gc.collect()
                        pbar.update(1)
                        continue

                    delay_s = k_dm * dm * geom_term
                    delay_samp = np.rint(delay_s / tsamp).astype(int)
                    max_delay = int(delay_samp.max())

                    n_read = patch_width + max_delay
                    if start + n_read > total_samples:
                        n_read = total_samples - start
                        if n_read < patch_width:
                            pass

                    block = self.filterbank_file.get_data(start, n_read).T
                    dedisp_patch = self._dedisperse_patch(block, dm)[:, :patch_width]
                    dedisp_patch = self.normalize_image_to_255(dedisp_patch)

                    if dedisp_patch.shape[1] < patch_width:
                        tmp = np.zeros((nchans, patch_width), dtype=np.uint8) #alternativ float16
                        tmp[:, :dedisp_patch.shape[1]] = dedisp_patch
                        dedisp_patch = tmp

                    shard_memmap[local_idx] = dedisp_patch

                    del block
                    del dedisp_patch
                    if local_idx % 256 == 0:
                        gc.collect()

                    pbar.update(1)

                try:
                    shard_memmap.flush()
                except Exception:
                    pass

                del shard_memmap
                shard_infos.append(
                    {
                        "index": shard_idx,
                        "path": os.path.basename(shard_path),
                        "start_sample": int(shard_start),
                        "end_sample": int(shard_end),
                        "rows": int(shard_size),
                    }
                )
                shard_idx += 1
                processed = shard_end

        manifest = {
            "dataset": "dedispersed_freq_time",
            "name": self.name_of_set,
            "dtype": "uint8",
            "patch_width": int(patch_width),
            "nchans": int(nchans),
            "total_samples": int(n_samples),
            "max_shard_size_bytes": int(self.max_shard_size_bytes),
            "shards": shard_infos,
        }
        manifest_path = os.path.join(
            self.freq_time_shard_dir,name,
            f"{self.name_of_set}_dedispersed_freq_time_manifest.json"
        )
        with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

        print(f"Saved {shard_idx} dedispersed shard(s) to {self.freq_time_shard_dir}/{name}")
        print(f"Dedispersed shard manifest written to {manifest_path}")



    def _dedisperse_patch(self, data, dm):
        """
        Dedispergiere einen freq-time Patch für eine gegebene DM.

        Args:
            data (np.ndarray): 2D-Array [nchans, nsamp] im Original-Filterbank-Frame.
            dm (float): Dispersion Measure in pc cm^-3.

        Returns:
            np.ndarray: dedispersed Patch [nchans, nsamp].
        """

        if not np.isfinite(dm) or dm == 0.0:
            return data.copy()

        hdr = self.filterbank_file.your_header
        nchans, nsamp = data.shape
        
        tsamp = float(hdr.tsamp)
        fch1 = float(hdr.fch1)                   # frequency in MHz
        foff = float(hdr.foff)               # MHz / Kanal (evtl. negativ)


        freqs = fch1 + np.arange(nchans) * foff  # MHz
        f_ref = np.max(freqs)                    # höchste Frequenz als Referenz

        # DM-Konstante (Lorimer & Kramer)
        k_dm = 4.148808e3  # s * MHz^2 * pc^-1 cm^3
        delay_s = k_dm * dm * (freqs**-2 - f_ref**-2)  # s
        delay_samp = np.rint(delay_s / tsamp).astype(int)

        dedisp = np.zeros_like(data, dtype=np.float32)

        for ch in range(nchans):
            s = data[ch] #maybe hier andere axis
            shift = int(delay_samp[ch])

            if shift > 0:
                # Kanal kommt später an -> nach links schieben
                dedisp[ch, :-shift] = s[shift:]
                dedisp[ch, -shift:] = 0.0
            elif shift < 0:
                # Kanal kommt früher an -> nach rechts schieben
                shift = -shift
                dedisp[ch, shift:] = s[:-shift]
                dedisp[ch, :shift] = 0.0
            else:
                dedisp[ch] = s

        return dedisp
    

        
            
    
    def _getDM(self, position, width=256):
        """
        Approximates DM, that TransientX would return for a Candidate (DM of max pixels in DM-Time).
        Has not to be exactly equal to TransientX method (using SNR Cubes created by Boxcar-Fitlers and DB-Clustering them),
        because it just returns a Trial-DM for Noise/RFI. It just shouldnt be DM=0 for every candidate.

        Args:
            position (int): startindex in time/sample space (columnindex in dm_time_image)
            width (int): amount timesteps of window

        Returns:
            float: Trial-DM 
        """
        n_dm, n_time = self.dm_time_image.shape

        # if window larger than image
        if width >= n_time:
            start_idx = 0
            end_idx = n_time
        else:
            start_idx = max(0, min(position, n_time - width))
            end_idx = start_idx + width

        patch = self.dm_time_image[:, start_idx:end_idx]

        #if only nans or empty
        if patch.size == 0 or not np.isfinite(patch).any():
            return float("nan")
        
        patch = np.where(np.isfinite(patch), patch, -np.inf)

        dm_idx, _ = np.unravel_index(np.argmax(patch), patch.shape)

        return float(self.dm_list[dm_idx])

