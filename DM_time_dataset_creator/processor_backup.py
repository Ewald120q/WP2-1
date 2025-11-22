import os
import glob
import numpy as np
import pandas as pd
import json
from decimal import Decimal
from tqdm import tqdm
import your


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
    def __init__(self, config_path, save_dedisp_freq_time = False):
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
        
        self.save_dedisp_freq_time = save_dedisp_freq_time

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

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
            dataset = np.empty([target_count, len(self.dm_list), 256], dtype=np.uint8)
            metadata = np.empty((target_count, 4), dtype=np.float64) # SNR, t_min, t_max, DM per sample
            global_index = 0
            
            # Use tqdm to display progress bar
            with tqdm(total=target_count, desc='Processing rest of events') as pbar:
                while global_index < target_count:
                    # Generate a random position
                    position = np.random.randint(0, self.dm_time_image.shape[1] - 256)
                    
                    # Check if position overlaps with pulses or artefacts
                    if not self._is_in_pulse_or_bbrfi(position, exclude_positions):
                        slice_start = position
                        slice_end = slice_start + 256
                        t_min, t_max = self._sample_index_to_time_bounds(slice_start, slice_end - slice_start)
                        dataset[global_index] = self.normalize_image_to_255(
                            self.dm_time_image[:, slice_start:slice_end][::-1]
                        )
                        dm = self.getDM(position)
                        metadata[global_index] = (np.nan, t_min, t_max, dm)
                        global_index += 1
                        pbar.update(1)  # Update progress bar
                
            return dataset, metadata
        else:
            # Process pulses or zero DM events
            dataset = np.empty([candidates.shape[0] * self.ntsamples, len(self.dm_list), 256], dtype=np.uint8)
            metadata = np.empty((candidates.shape[0] * self.ntsamples, 4), dtype=np.float64)
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
                    dataset[global_index] = self.normalize_image_to_255(
                        self.dm_time_image[:, slice_start:slice_end][::-1]
                    )
                    metadata[global_index] = (float(row['snr']), t_min, t_max, float(row['dm']))
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


    def process(self):
        """
        Main function to process candidates and create the DM-Time dataset.

        Steps:
            1. Load and categorize candidates.
            2. Process each category ('pulses', 'zero DM events', 'rest of events').
            3. Combine datasets, shuffle, and save as .npy files.

        Outputs:
            - Combined dataset as a .npy file.
            - Corresponding labels as a .npy file.
        """
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
        dataset_with_pulses, metadata_pulses = self._process_candidates(pulse_candidates, 'pulses')
        dataset_with_bbrfi, metadata_bbrfi = self._process_candidates(zero_dm_events, 'zero DM events')
        
        target_count = dataset_with_pulses.shape[0] + dataset_with_bbrfi.shape[0]

        # Generate rest of events
        dataset_with_rest, metadata_rest = self._process_candidates(
            rest, 
            'rest of events', 
            exclude_positions=exclude_positions, 
            target_count=target_count
)
    
        # Combine datasets and create labels
        combined_data = np.concatenate((dataset_with_pulses, dataset_with_bbrfi, dataset_with_rest), axis=0)
        combined_metadata = np.concatenate((metadata_pulses, metadata_bbrfi, metadata_rest), axis=0)
        labels = np.array(
            ['Pulse'] * len(dataset_with_pulses) +
            ['Artefact'] * (len(dataset_with_bbrfi) + len(dataset_with_rest))
        )
    
        # Shuffle data and labels
        indices = np.random.permutation(len(combined_data))
        shuffled_data = combined_data[indices]
        shuffled_labels = labels[indices]
        shuffled_metadata = combined_metadata[indices]
    
        # Save final datasets
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased.npy'), shuffled_data)
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased_labels.npy'), shuffled_labels)
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased_metadata.npy'), shuffled_metadata)

    def report_snr_statistics(self, metadata_path=None, labels_path=None, snr_round=2):
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
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")

        metadata = np.load(metadata_path)
        labels = np.load(labels_path)

        if metadata.ndim != 2 or metadata.shape[1] < 3:
            raise ValueError(
                "Expected metadata with shape (N, 3) where columns are (snr, t_min, t_max)."
            )
        if labels.shape[0] != metadata.shape[0]:
            raise ValueError(
                f"Label count ({labels.shape[0]}) does not match metadata count ({metadata.shape[0]})."
            )

        snr_values = metadata[:, 0]
        total_samples = labels.shape[0]
        finite_mask = np.isfinite(snr_values)
        finite_snr = snr_values[finite_mask]

        print("=== Dataset Overview ===")
        print(f"Total samples: {total_samples}")
        print(f"Samples with finite SNR: {finite_snr.size} ({finite_snr.size / total_samples:.2%})")
        print(f"Samples with NaN/inf SNR: {total_samples - finite_snr.size}")

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

        pulse_mask = labels == 'Pulse'
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
            
    
    def getDM(self, position, width=256):
        """
        Approximiert die DM, die TransientX für einen Kandidaten zurückgeben würde (DM des maximalen Pixels in DM-time),
        Muss nicht exakt identisch zu TransientX sein (mit wilden SNR Cubes durch Boxcar-Filter und anschließendes DB-Clustering),
        da hiermit eh nur eine Trial-DM für Noise/RFI zurückgegeben wird.

        Args:
            position (int): Startindex im Zeit/Sample-Raum (Spaltenindex in dm_time_image).
            width (int): Anzahl der Zeitsamples, die das Fenster umfasst.

        Returns:
            float: Trial-DM 
        """
        n_dm, n_time = self.dm_time_image.shape

        # falls fenster größer als bild
        if width >= n_time:
            start_idx = 0
            end_idx = n_time
        else:
            start_idx = max(0, min(position, n_time - width))
            end_idx = start_idx + width

        patch = self.dm_time_image[:, start_idx:end_idx]

        #falls nur nan oder leer
        if patch.size == 0 or not np.isfinite(patch).any():
            return float("nan")
        
        patch = np.where(np.isfinite(patch), patch, -np.inf)

        dm_idx, _ = np.unravel_index(np.argmax(patch), patch.shape)

        return float(self.dm_list[dm_idx])

