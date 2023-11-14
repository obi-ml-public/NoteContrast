# Python code to create shell script files that will run the python scripts
# to generate the data required for training.

import glob

python_script = 'python /mnt/obi0/pk621/projects/icd_embeddings/dataset/setup/icd_dataset.py --input_file {} --output_folder /mnt/obi0/phi/ehr_projects/icd_embeddings/data/icd10_sequences_v1/'
input_folder = '/mnt/obi0/phi/ehr_projects/ehr_active/data/icd10_pretraining/ecg_hem_vir/encounters/'
output_folder = '/mnt/obi0/pk621/projects/icd_embeddings/run/dataset/icd_dataset_partition/'

all_files = [file for file in glob.glob(input_folder + '*parquet')]
total_files_count = len(all_files)

number_of_chunks = 10
chunk_size = total_files_count // number_of_chunks

for index in range(number_of_chunks + 1):
    lower_index = index * chunk_size
    upper_index = (index + 1) * chunk_size

    current_chunk = all_files[lower_index:upper_index]

    python_scripts = [python_script.format(input_file) for input_file in current_chunk]

    with open(output_folder + f'{index}.sh', 'w') as file:
        for script in python_scripts:
            file.write(script + '\n\n')