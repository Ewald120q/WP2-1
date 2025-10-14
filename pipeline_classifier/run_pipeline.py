import subprocess
import argparse
import os

from utils import run_command, create_buffer, kill_buffer, load_config, kill_dada_dbdedispdb, kill_dada_fildb

parser = argparse.ArgumentParser(description='Bowtie recognition pipline')
parser.add_argument('-c', '--config', type=str, required=True, help='Config file')

args = parser.parse_args()
    
name_of_config = os.path.splitext(os.path.basename(args.config))[0]
config = load_config(args.config)



# 1. Creating buffers
create_buffer(config['path_to_pulsarx_singularity_image'], buffer_size=config['input_buffer_size'], num_blocks=16, key=config['key_input'])
create_buffer(config['path_to_pulsarx_singularity_image'], buffer_size=config['input_buffer_size']*16, num_blocks=16, key=config['key_output'])

# 2. Running fildb and dbdedispdb
commands = [
    f'singularity exec -B $PWD -B {config["path_to_filterbanks"]} {config["path_to_pulsarx_singularity_image"]} dada_fildb --key_output {config["key_input"]} -f {config["path_to_filterbanks"]}{config["name_of_the_filterbank"]}',
    f'singularity exec -B $PWD {config["path_to_pulsarx_singularity_image"]} dada_dbdedispdb --key_input {config["key_input"]} --key_output {config["key_output"]} -c {config["path_to_a_transientx_config"]} -t {config["threads_for_transientx"]}'
]

# Execute commands without waiting
for command in commands:
    run_command(command, wait=False)
    

# 3. Running run_inference.py and waiting for it to finish
pytorch_inference_command = f'singularity exec -B $PWD -B {config["path_to_models"]} {config["path_to_tensorflow_psrdada_singularity_image"]} python3 run_inference.py -c {args.config}'
### 3. Running run_inference.py and waiting for it to finish (using local Python environment)
###pytorch_inference_command = f'python run_inference.py -c {args.config}'
run_command(pytorch_inference_command, wait=True)


# 4. Killing proceses
kill_dada_dbdedispdb(config['key_input'], config['key_output'])
kill_dada_fildb(config['key_input'])

kill_buffer(config['path_to_pulsarx_singularity_image'], config['key_input'])
kill_buffer(config['path_to_pulsarx_singularity_image'], config['key_output'])

print('All commands have been executed.')
