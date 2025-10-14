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

    def _process_candidates(self, candidates, label, exclude_positions=None, target_count=None):
        """
        Process candidates to extract data or generate random segments for the 'rest' category.

        Args:
            candidates (pd.DataFrame): DataFrame of candidates to process.
            label (str): Category label ('pulses', 'zero DM events', or 'rest of events').
            exclude_positions (list, optional): List of position ranges to exclude for 'rest of events'.
            target_count (int, optional): Target number of samples for 'rest of events'.

        Returns:
            numpy.ndarray: Extracted or generated dataset for the given category.
        """
        if label == 'rest of events':
            # Ensure target count is provided
            if target_count is None:
                raise ValueError("Target count must be specified for 'rest of events'")
    
            # Initialize dataset with the target count
            dataset = np.empty([target_count, len(self.dm_list), 256], dtype=np.uint8)
            global_index = 0
            
            # Use tqdm to display progress bar
            with tqdm(total=target_count, desc='Processing rest of events') as pbar:
                while global_index < target_count:
                    # Generate a random position
                    position = np.random.randint(0, self.dm_time_image.shape[1] - 256)
                    
                    # Check if position overlaps with pulses or artefacts
                    if not self._is_in_pulse_or_bbrfi(position, exclude_positions):
                        dataset[global_index] = self.normalize_image_to_255(
                            self.dm_time_image[:, position:position + 256][::-1]
                        )
                        global_index += 1
                        pbar.update(1)  # Update progress bar
                
            return dataset
        else:
            # Process pulses or zero DM events
            dataset = np.empty([candidates.shape[0] * self.ntsamples, len(self.dm_list), 256], dtype=np.uint8)
            global_index = 0
    
            for idx, row in tqdm(candidates.iterrows(), total=candidates.shape[0], desc=f'Processing {label}'):
                position = int(self._get_position_in_filfile(row['mjd']))
                for i in range(self.ntsamples):
                    dataset[global_index] = self.normalize_image_to_255(
                        self.dm_time_image[:, position - i:position - i + 256][::-1]
                    )
                    global_index += 1
    
            return dataset



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
        dataset_with_pulses = self._process_candidates(pulse_candidates, 'pulses')
        dataset_with_bbrfi = self._process_candidates(zero_dm_events, 'zero DM events')
        
        target_count = len(dataset_with_pulses) + len(dataset_with_bbrfi)

        # Generate rest of events
        dataset_with_rest = self._process_candidates(
            rest, 
            'rest of events', 
            exclude_positions=exclude_positions, 
            target_count=target_count
)
    
        # Combine datasets and create labels
        combined_data = np.concatenate((dataset_with_pulses, dataset_with_bbrfi, dataset_with_rest), axis=0)
        labels = np.array(
            ['Pulse'] * len(dataset_with_pulses) +
            ['Artefact'] * (len(dataset_with_bbrfi) + len(dataset_with_rest))
        )
    
        # Shuffle data and labels
        indices = np.random.permutation(len(combined_data))
        shuffled_data = combined_data[indices]
        shuffled_labels = labels[indices]
    
        # Save final datasets
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased.npy'), shuffled_data)
        np.save(os.path.join(self.output_dir, f'{self.name_of_set}_DM_time_dataset_realbased_labels.npy'), shuffled_labels)