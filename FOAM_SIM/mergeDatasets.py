
import os
import shutil
import pandas as pd
import re

def combine_folders(folder_a, folder_b, output_folder):

    fields_output = os.path.join(output_folder, 'fields')
    probes_output = os.path.join(output_folder, 'probes')
    os.makedirs(fields_output, exist_ok=True)
    os.makedirs(probes_output, exist_ok=True)

    def copy_and_rename_files(src_folder, dest_folder):

        for file_name in sorted(os.listdir(src_folder)):
            src_file_path = os.path.join(src_folder, file_name)
            if os.path.isfile(src_file_path):
                match = re.search(r'(\d+)', file_name)
                if match:
                    base_name, ext = os.path.splitext(file_name)
                    original_number = int(match.group(1))
                    new_number = original_number

                    while True:
                        new_file_name = re.sub(r'(\d+)', f'{new_number:03d}', base_name) + ext
                        new_file_path = os.path.join(dest_folder, new_file_name)
                        if not os.path.exists(new_file_path):
                            break
                        new_number += 1

                    shutil.copy(src_file_path, new_file_path)
                else:
                    shutil.copy(src_file_path, os.path.join(dest_folder, file_name))
                    
    def merge_csv_files(file_a, file_b, output_file):
        
        if os.path.exists(file_a) and os.path.exists(file_b):
            df_a = pd.read_csv(file_a)
            df_b = pd.read_csv(file_b)
            combined_df = pd.concat([df_a, df_b], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
        elif os.path.exists(file_a):
            shutil.copy(file_a, output_file)
        elif os.path.exists(file_b):
            shutil.copy(file_b, output_file)

    print("Processing fields...")
    copy_and_rename_files(os.path.join(folder_a, 'fields'), fields_output)
    copy_and_rename_files(os.path.join(folder_b, 'fields'), fields_output)

    print("Processing probes...")
    copy_and_rename_files(os.path.join(folder_a, 'probes'), probes_output)
    copy_and_rename_files(os.path.join(folder_b, 'probes'), probes_output)

    print("Merging simulation_parameters.csv...")
    simulation_params_a = os.path.join(folder_a, 'simulation_parameters.csv')
    simulation_params_b = os.path.join(folder_b, 'simulation_parameters.csv')
    output_simulation_params = os.path.join(output_folder, 'simulation_parameters.csv')
    merge_csv_files(simulation_params_a, simulation_params_b, output_simulation_params)

    print("Copying checkpoints folder...")
    checkpoints_src = os.path.join(folder_a, 'checkpoints')
    checkpoints_dest = os.path.join(output_folder, 'checkpoints')
    if os.path.exists(checkpoints_src):
        shutil.copytree(checkpoints_src, checkpoints_dest, dirs_exist_ok=True)

    print("Copying connectivity.npy...")
    connectivity_src = os.path.join(folder_a, 'connectivity.npy')
    connectivity_dest = os.path.join(output_folder, 'connectivity.npy')
    if os.path.exists(connectivity_src):
        shutil.copy(connectivity_src, connectivity_dest)

    print(f"All files successfully processed and combined into {output_folder}")


path = "C:/Users/crist/Documents/GitHub/CFD-DL/.data/"
folder_a = path + "Dataset_50sims"
folder_b = path + "Extracted_data"
output_folder = path + "Dataset_75sims"

combine_folders(folder_a, folder_b, output_folder)
