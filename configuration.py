import json
import csv
import utils
import pandas as pd
import os
import keyboard  # type: ignore
import glob
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings("ignore")

def load_configuration(config_file):
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Configuration file not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON in configuration file: {e}")
        return None


# def map_features(configuration, preconfigured_file=None):
#     mapping = {}

#     if preconfigured_file:
#         # Load preconfigured mappings from the file
#         with open(preconfigured_file, 'r') as file:
#             preconfigured_mapping = json.load(file)
        
#         if 'column_mapping' in preconfigured_mapping:
#             for target_feature, source_feature in preconfigured_mapping['column_mapping'].items():
#                 if source_feature:  # Only add to mapping if source_feature is not empty
#                     mapping[target_feature] = source_feature
            
#             print("Using preconfigured mappings:")
#             for target_feature, source_feature in mapping.items():
#                 print(f"'{target_feature}' is mapped to '{source_feature}'")
#         else:
#             print("Invalid preconfigured file format. Missing 'column_mapping'.")
#     else:
#         # Interactive mapping
#         print("Please map the features (enter 'skip' to omit a feature):")
#         for target_feature in configuration['column_mapping']:
#             source_feature = input(f"Map '{target_feature}' to source feature (or 'skip'): ")
#             if source_feature.lower() != 'skip' and source_feature != '':
#                 mapping[target_feature] = source_feature
#             else:
#                 print(f"Skipping mapping for '{target_feature}'.")

#         # Allow the user to add additional mappings
#         while True:
#             print("Do you want to add mappings for additional features? (yes/no)")
#             response = input().lower()
#             if response == 'no':
#                 break
#             elif response == 'yes':
#                 additional_feature = input("Enter the name of the additional feature in the input file: ")
#                 mapped_feature = input("What should this map to? ")
#                 mapping[additional_feature] = mapped_feature
#             else:
#                 print("Invalid response, please enter 'yes' or 'no'.")

#     return mapping

import json

def map_features(configuration, source_columns, preconfigured_file=None):
    mapping = {}

    if preconfigured_file:
        # Load preconfigured mappings from the file
        with open(preconfigured_file, 'r') as file:
            preconfigured_mapping = json.load(file)
        
        if 'column_mapping' in preconfigured_mapping:
            for target_feature, source_feature in preconfigured_mapping['column_mapping'].items():
                if source_feature:  # Only add to mapping if source_feature is not empty
                    mapping[target_feature] = source_feature
            
            print("Using preconfigured mappings:")
            for target_feature, source_feature in mapping.items():
                print(f"'{target_feature}' is mapped to '{source_feature}'")
        else:
            print("Invalid preconfigured file format. Missing 'column_mapping'.")
    else:
        # Interactive mapping
        print("Please map the features (enter 'skip' to omit a feature):")
        for target_feature, properties in configuration['column_mapping'].items(): 
            while True:
                source_feature = input(f"Map '{target_feature}' to source feature (or 'skip'):\n> ")
                if source_feature.lower() == 'skip' or source_feature == '':
                    if properties['mandatory']:
                        print(f"Feature '{target_feature}' is mandatory and cannot be skipped. Please map it.")
                        continue  # Forces the user to provide a valid mapping for mandatory features.
                    else:
                        print(f"Skipping mapping for '{target_feature}'.")
                        break  
                elif source_feature in source_columns:
                    mapping[target_feature] = source_feature
                    break
                else:
                    print(f"'{source_feature}' does not exist in the source columns. Please re-map '{target_feature}'.")

        # Allow the user to add additional mappings
        while True:
            print("Do you want to add mappings for additional features? (yes/no)")
            response = input().lower()
            if response == 'no':
                break
            elif response == 'yes':
                while True:
                    additional_feature = input("Enter the name of the additional feature in the input file:\n> ")
                    mapped_feature = input("What should this map to?\n> ")
                    if mapped_feature in source_columns:
                        mapping[additional_feature] = mapped_feature
                        break
                    else:
                        print(f"'{mapped_feature}' does not exist in the source columns. Please re-map the feature.")
            else:
                print("Invalid response, please enter 'yes' or 'no'.")

    return mapping

def get_unique_filename(directory, filename):
    """
    Returns a unique filename by appending a number to the filename if it already exists.
    """
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1

    return new_filename

def select_folder_to_save():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_selected = filedialog.askdirectory(title="Select Directory to Save CSV Files")

    return folder_selected

def save_dfs_as_csv(dataframes):
    # Create a Tkinter root window
    print('Select the directory where you want to save the files.')
    
    save_dir = select_folder_to_save()
    
    # Ensure the directory exists (if not selected, use current directory)
    if not save_dir:
        save_dir = os.getcwd()

    # Save each DataFrame as a CSV file in the specified directory
    for i, df in enumerate(dataframes):
        filename = f"dataframe_{i+1}.csv"
        save_path = os.path.join(save_dir, filename)

        # Get a unique filename to avoid overwriting existing files
        unique_filename = get_unique_filename(save_dir, filename)
        unique_save_path = os.path.join(save_dir, unique_filename)

        try:
            df.to_csv(unique_save_path, index=False, sep=';')
            print(f"DataFrame {i+1} successfully saved to {unique_save_path}")
        except Exception as e:
            print(f"Error saving DataFrame {i+1} to {unique_save_path}: {e}")



def process_csv(input_file, output_file, feature_mapping):
    with open(input_file, 'r', encoding= 'utf-8-sig') as input_csv, open(output_file, 'w', newline='') as output_csv:
        reader = csv.DictReader(input_csv, delimiter=';')
        writer = csv.DictWriter(output_csv, fieldnames=feature_mapping.keys(), delimiter=';')
        writer.writeheader()
        for row in reader:
            mapped_row = {target_feature: row[source_feature] for target_feature, source_feature in feature_mapping.items()}
            writer.writerow(mapped_row)
    df = pd.read_csv(output_file, delimiter = ';')
    return df

def load_source_columns(file_path):
    df = pd.read_csv(file_path,sep = ';', nrows=0)
    return df.columns.tolist()

def config(config_file):
    while True:
        print('Do you wish to select a single file or a folder with files to configurate:\n a) Single File\n b) Folder\n c) Go back\n')
        file_or_folder = input('>')
        if file_or_folder == 'a':
            print('Select the file:')
            input_path = file_path()
        elif file_or_folder == 'b':
            print('Select the folder')
            input_path = folder_path()
        elif file_or_folder == 'c':
            print('Going back...')
            break
        
        configuration = load_configuration(config_file)
        mapping_style = input("Do you wish to manually map the features or do you want to load a configuration file?\n a) Manually\n b) Configuration File\n> ")

        source_columns = load_source_columns(input_path[0])
        
        if mapping_style == 'a':
            feature_mapping = map_features(configuration, source_columns)
        elif mapping_style == 'b':
            
            print("Select the preconfigured file: ")
            configuration_file = file_path()
            feature_mapping = map_features(configuration, source_columns, configuration_file)
        else:
            print('Choose a correct option - a or b!')
            return
        
        print("Select the output folder: ")
        output_folder = select_folder_to_save()
        print(f'This is the output folder: {output_folder} and this its type {type(output_folder)}')
        

        processed_files = {}
        
        for input_file in input_path:
            output_file = os.path.join(output_folder, os.path.basename(input_file))
            df = process_csv(input_file, output_file, feature_mapping)
            print(f"CSV conversion for {input_file} completed successfully.")
            processed_files[input_file] = df

       
    
        return processed_files

def file_path():
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename()  # Open the file dialog
        return [file_path]
    except tk.TclError:
        print("Exiting...")
        exit()

def folder_path():
    print('\nSelect the folder path in the file explorer window.')
    """Opens a folder dialog for the user to select a folder and returns the paths of all files inside the folder, excluding .DS_Store."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory()  # Open the folder dialog
    if folder_path:
        file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, file)) and file != '.DS_Store']
        return file_paths
    return []
    

def merge_files_or_folder(merged_files):
    while True:
        print('Do you wish to join a single file or a folder?')
        directory_choice = input('a) Single file\nb) Folder\nc) Go back\n>')
        if directory_choice == 'a': 
            while True:
                other = file_path()
                other = load_dataframes(other)
                all_valid = True
                for df in other:
                    if 'tempo(s)' not in df.columns:
                        print('The files you selected do not have a column called "tempo(s)". Please select the files again.')
                        all_valid = False
                        break
                if all_valid:
                    break
            merged_files = utils.join_other(merged_files, other)
            save_dfs_as_csv(merged_files)
            break
        elif directory_choice == 'b':
            while True:
                other = folder_path()
                other = load_dataframes(other)
                all_valid = True
                for df in other:
                    if 'tempo(s)' not in df.columns:
                        print('The files you selected do not have a column called "tempo(s)". Please select the files again.')
                        all_valid = False
                        break
                if all_valid:
                    break
            merged_files = utils.join_other(merged_files, other)
            save_dfs_as_csv(merged_files)
            break
        elif directory_choice == 'c':
            break
        else: 
            print('Choose a valid option - a, b or c!')
            break


def prepare_eyetracker(list_eye):
    list_eye = [utils.impute_gaze_mv(item) for item in list_eye]
    list_eye = [utils.outlier_perclos(item, 'replace') for item in list_eye]
    list_eye = [utils.create_open_eye(item) for item in list_eye]
    return list_eye



def load_dataframes(input_files):
    dataframes = [pd.read_csv(file, delimiter=';', engine='python') for file in input_files]
    return dataframes

def select_files_option_b(device):
    file_or_folder = input('Do wish to select a single file or a folder?\na) File\nb) Folder\nc) Go back\n>')
    while True:
        if device == 'simulator':
            if file_or_folder == 'a':
                print('\nVehicle simulator File Selection (Velocity, etc.)')
                sim_txt_files = file_path()
                #read sim_txt_files as csv
                print('\nParticipant File Selection (KSS, etc.)')
                sim_log_files = file_path()
                
            elif file_or_folder == 'b':
                print('\nVehicle simulator Folder Selection (Velocity, etc.)')
                sim_txt_files = folder_path()
                #read sim_txt_files as csv
                print('\nParticipant Folder Selection (KSS, etc.)')
                sim_log_files = folder_path()
                
            elif file_or_folder == 'c':
                print('Going back...')
                break
            else:
                print('Select a valid option - a, b or c!')
                continue
            return sim_txt_files, sim_log_files
        elif device == 'smartwatch':
            if file_or_folder == 'a':
                print('\nSmartwatch File Selection:')
                smartwatch = file_path()
            elif file_or_folder == 'b':
                print('\nSmartwatch Folder Selection (Velocity, etc.)')
                smartwatch = folder_path()
            elif file_or_folder == 'c':
                print('Going back...')
                break
            else:
                print('Select a valid option - a, b or c!')
                continue
            return smartwatch
        elif device == 'eyetracker':
            if file_or_folder == 'a':
                print('\nLeft Gaze File Selection:')
                left_gaze = file_path()
                #read sim_txt_files as csv
                print('\nRight Gaze File Selection')
                right_gaze = file_path()
                print('\nVergence Gaze File Selection')
                ver_gaze = file_path()
            elif file_or_folder == 'b':
                print('\nLeft Gaze Folder Selection:')
                left_gaze = folder_path()
                #read sim_txt_files as csv
                print('\nRight Gaze Folder Selection')
                right_gaze = folder_path()
                print('\nVergence Gaze Folder Selection')
                ver_gaze = folder_path()
            elif file_or_folder == 'c':
                print('Going back...')
                return None
            else:
                print('Select a valid option - a, b or c!')
                continue
            return left_gaze, right_gaze, ver_gaze




def main():
    while True:
        # if keyboard.is_pressed('esc'):
        #     print("Script stopped by pressing 'Esc' key.")
        #     break
        version = input('Choose version:\na) Extract and join data in the BBAI format\nb) Join previously mapped files\nc) Map/Convert files\nd) Exit\n> ')
        if version == 'a':
            print('You have chosen option a.\n Choose the experiment you want to extract (drowsiness or distraction).')
            #a função do utils deve ser alterada de forma a que se possa escolher o path
            while True:
                
                experiment = input('Choose the experiment you want to extract (drowsiness or distraction).\na) Drowsiness\nb) Distraction\nc) Go back\n>')    
                if experiment == 'a' or experiment == 'b': #depois tem de haver uma distinção entre os dois
                    if experiment == 'a':
                        experiment = 'drowsiness'
                        # print("You've chosen the drowsiness experiment. Which files do you want to extract?\n")
                        # files_to_extract = input('\n a) Simulator\n b) Simulator + Smartwatch \n c) Simulator + Eyetracker \n d) Simulator + Eyetracker + Smartwatch\n e) Combinations with other files\n f) Go back\n>')
                        while True:
                            print("You've chosen the drowsiness experiment. Which files do you want to extract?\n")
                            files_to_extract = input('\n a) Simulator\n b) Simulator + Smartwatch \n c) Simulator + Eyetracker \n d) Simulator + Eyetracker + Smartwatch\n e) Combinations with other files\n f) Go back\n>')
                        
                            if files_to_extract == 'a':
                                try: 
                                    merged_files = utils.extract_simulator(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                                except FileNotFoundError as e:
                                    print(f"Error: {e}")
                                    #base_path = utils.select_base_path()  # Ask user to select the base path again
                                    merged_files = utils.extract_simulator(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                                except ValueError as e:
                                    print("You probably selected the wrong file. Please try again.")
                                    #base_path = utils.select_base_path()  # Ask user to select the base path again
                                    merged_files = utils.extract_simulator(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                                
                            elif files_to_extract == 'c':
                                try:
                                    merged_files = utils.extract_sim_eyetracker(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                                except ValueError as e:
                                    print("You probably selected the wrong file. Please try again.")
                                    #base_path = utils.select_base_path()  # Ask user to select the base path again
                                    merged_files = utils.extract_sim_eyetracker(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                            elif files_to_extract == 'b':
                                try:
                                    merged_files = utils.extract_sim_smartwatch(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                                except ValueError as e: 
                                    print('You probably selected the wrong file. Please try again.')
                                    merged_files = utils.extract_sim_smartwatch(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return

                            elif files_to_extract == 'd':
                                try:
                                    merged_files = utils.extract_sim_sm_eyet(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                                except ValueError as e:
                                    print('You probably selected the wrong file. Please try again.')
                                    merged_files = utils.extract_sim_sm_eyet(experiment)
                                    save_dfs_as_csv(merged_files)
                                    return
                            
                            elif files_to_extract == 'e':
                                while True:
                                    print('Choose one of the following options:')
                                    other_options = input('\n a) Simulator + Others\n b) Simulator + Smartwatch + Others \n c) Simulador + Eye Tracker + Others\n d) Simulator + Eyetracker + Smartwatch + Others\n e) Go back\n>')
                                
                                    if other_options == 'a':
                                        merged_files = utils.extract_simulator(experiment)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_files)

                                    elif other_options == 'c':
                                        merged_files = utils.extract_sim_eyetracker(experiment)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_files)
                                    elif other_options == 'b':
                                        merged_files = utils.extract_sim_smartwatch(experiment)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_files)

                                    elif other_options == 'd':
                                        merged_files = utils.extract_sim_sm_eyet(experiment)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_files)
                                    elif other_options == 'e':
                                        print('Exiting...')
                                        break
                                    else:
                                        print('Choose a valid option - a, b, c, d or e')
                                        continue
                            elif files_to_extract == 'f':
                                print('Exiting...')
                                break
                            else:
                                print('Choose a valid option - a, b, c, d, e or f!')
                                continue

                    elif experiment == 'b':
                        experiment = 'distraction'
                        while True:
                            print("You've chosen the distraction experiment. Which files do you want to extract?\n")
                            files_to_extract = input('\n a) Simulator\n b) Simulator + Smartwatch \n c) Simulator + Eye Tracker \n d) Simulator + Eyetracker + Smartwatch\n e) Combinations with other files \n f) Go back\n>')
                            if files_to_extract == 'a':
                                merged_files = utils.extract_simulator(experiment)
                                save_dfs_as_csv(merged_files)
                                return
                                
                            elif files_to_extract == 'c':
                                merged_files = utils.extract_sim_eyetracker(experiment)
                                save_dfs_as_csv(merged_files)
                                return
                            elif files_to_extract == 'b':
                                merged_files = utils.extract_sim_smartwatch(experiment)
                                save_dfs_as_csv(merged_files)
                                return

                            elif files_to_extract == 'd':
                                merged_files = utils.extract_sim_sm_eyet(experiment)
                                save_dfs_as_csv(merged_files)
                                return

                            elif files_to_extract == 'e':
                                while True:
                                    print('Choose one of the following options:')
                                    other_options = input('\n a) Simulador + Others\n b) Simulador + Smartwatch + Others \n c) Simulador + Eye Tracker + Others\n d) Simulador + Eyetracker + Smartwatch + Others\n e) Go back\n>')
                                    if other_options == 'a':
                                        merged_files = utils.extract_simulator(experiment)
                                        merge_files_or_folder(merged_files)
                                    
                                    elif other_options == 'c':
                                        merged_files = utils.extract_sim_eyetracker(experiment)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_files)
                                    elif other_options == 'b':
                                        merged_files = utils.extract_sim_smartwatch(experiment)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_files)

                                    elif other_options == 'd':
                                        merged_files = utils.extract_sim_sm_eyet(experiment)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_files)
                                    
                                    elif other_options == 'e': 
                                        print('Exiting...')
                                        break
                                    else:
                                        print('Choose a valid option - a, b, c, d or e')



                            elif files_to_extract == 'f':
                                print('Exiting...')
                                break
                            else:
                                print('Choose a valid option.')
                            
                        
                elif experiment == 'c':
                    print('Going back...')
                    break
                else:
                    print('Choose a valid option - a ou b')      
        elif version == 'b': 
            print('You have chosen option b.\n')
            while True:
                experiment = input(' Choose the experiment you want to extract (drowsiness or distraction).\na) Drowsiness\nb) Distraction\nc) Go back\n>')
                if experiment == 'a':
                    experiment = 'drowsiness'
                    join_option_b = input('Which ones do you want to join?\n a) Simulator Files\n b) Simulator + Eyetracker Files\n c) Simulator + Smartwatch Files\n d) Simulator + Smartwatch + Eyetracker Files\n e) Combinations with other files\n f) Go back\n>')
                    
                    
                    while True:
                        if join_option_b == 'a': 
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            if sim_txt_files and sim_log_files:
                            
                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files)
                                merged_sim = utils.merge_sims(sim_txt_csvs, sim_log_csvs)
                                save_dfs_as_csv(merged_sim)
                                return
                            else:
                                print("Error selecting the files.")
                        elif join_option_b == 'b':
                            
                            left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                                gaze_left_csvs = load_dataframes(left_gaze_files)
                                gaze_right_csvs = load_dataframes(right_gaze_files)
                                gaze_ver_csvs = load_dataframes(ver_gaze_files)
                                gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                                gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                                gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                                gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                                while True:
                                    user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                    if user_input.lower() == 'yes':
                                        print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                        sinc_path = file_path()
                                        gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                        break
                                    elif user_input.lower() == 'no':
                                        break
                                    else:
                                        print("Please enter 'yes' or 'no'.")
                                        continue

                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files) 
                                merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                                merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample') 
                                save_dfs_as_csv(merged_sim_eyet)  
                                return          
                            else:
                                print('Error selecting the files')
                        elif join_option_b == 'c': 
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            print('\n Smartwatch File Selection:')
                            smartwatch = select_files_option_b('smartwatch')
                            if  sim_txt_files and sim_log_files:     
                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files) 
                                smartwatch_dfs = load_dataframes(smartwatch) 
                                smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                                while True:
                                    user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                    if user_input.lower() == 'yes':
                                        print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                        sinc_path = file_path()
                                        smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch', sinc_path)
                                        break
                                    elif user_input.lower() == 'no':
                                        break
                                    else:
                                        print("Please enter 'yes' or 'no'.")
                                        continue
                                
                                merged_sim = utils.merge_sims(sim_txt_csvs,sim_log_csvs)
                                merged_sim_sw = utils.join_sim_w_smartwatch(merged_sim, smartwatch_dfs)
                                save_dfs_as_csv(merged_sim_sw)
                                return
                            else:
                                print('Error selecting the files')
                                continue
                        elif join_option_b == 'd':
                            left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            smartwatch = select_files_option_b('smartwatch')
                            if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                                gaze_left_csvs = load_dataframes(left_gaze_files)
                                gaze_right_csvs = load_dataframes(right_gaze_files)
                                gaze_ver_csvs = load_dataframes(ver_gaze_files)
                                gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                                gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                                gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                                gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                                while True:
                                    user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                    if user_input.lower() == 'yes':
                                        print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                        sinc_path = file_path() 
                                        gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)    
                                        break
                                    elif user_input.lower() == 'no':
                                        break
                                    else:
                                        print("Please enter 'yes' or 'no'.")
                                        continue
                                    
                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files)  
                                merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                                merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample')  
                                smartwatch_dfs = load_dataframes(smartwatch) 
                                smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                                while True:
                                    user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                    if user_input.lower() == 'yes':
                                        print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                        sinc_path = file_path()
                                        smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch', sinc_path)
                                        break
                                    elif user_input.lower() == 'no':
                                        break
                                    else:
                                        print("Please enter 'yes' or 'no'.")
                                        continue
                                
                                merged_full = utils.join_smartwatch_txt(merged_sim_eyet, smartwatch_dfs, 'other')
                                save_dfs_as_csv(merged_full)
                                return
                            else: 
                                print('Error selecting the files')
                                continue
                        elif join_option_b == 'e': 
                            
                            while True:
                                print('Choose one of the following options:')
                                other_options = input('\n a) Simulator + Others\n b) Simulator + Eye Tracker + Others \n c) Simulador + Smartwatch + Others\n d) Simulator + Eyetracker + Smartwatch + Others\n e) Go back\n>')

                                if other_options == 'a':
                                    sim_txt_files, sim_log_files = select_files_option_b('simulator')
                                    if sim_txt_files and sim_log_files:
                                        sim_txt_csvs = load_dataframes(sim_txt_files)
                                        sim_log_csvs = load_dataframes(sim_log_files)
                                        merged_sim = utils.merge_sims(sim_txt_csvs, sim_log_csvs)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_sim) 
                                        continue
                                    else:
                                        print("Error selecting the files.")
                                        
                                        
                                elif other_options == 'b':
                                    sim_txt_files, sim_log_files = select_files_option_b('simulator')
                                    left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                                
                                    if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                                        gaze_left_csvs = load_dataframes(left_gaze_files)
                                        gaze_right_csvs = load_dataframes(right_gaze_files)
                                        gaze_ver_csvs = load_dataframes(ver_gaze_files)
                                        gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                                        gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                                        gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                                        gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                                        while True:
                                            user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                            if user_input.lower() == 'yes':
                                                print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                                sinc_path = file_path()
                                                gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                                break
                                            elif user_input.lower() == 'no':
                                                break
                                            else:
                                                print("Please enter 'yes' or 'no'.")
                                                continue
                                            
                                        sim_txt_csvs = load_dataframes(sim_txt_files)
                                        sim_log_csvs = load_dataframes(sim_log_files) 
                                        merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                                        merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample')
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible') 
                                        merge_files_or_folder(merged_sim_eyet)  
                                        continue          
                                    else:
                                        print('Error selecting the files')#estou aqui

                                    
                                elif other_options == 'c':
                                    sim_txt_files, sim_log_files = select_files_option_b('simulator')
                                    print('\n Smartwatch File Selection:')
                                    smartwatch = select_files_option_b('smartwatch')
                                    if  sim_txt_files and sim_log_files:     
                                        sim_txt_csvs = load_dataframes(sim_txt_files)
                                        sim_log_csvs = load_dataframes(sim_log_files) 
                                        smartwatch_dfs = load_dataframes(smartwatch) 
                                        smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                                        while True:
                                            user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                            if user_input.lower() == 'yes':
                                                print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                                sinc_path = file_path()
                                                smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch', sinc_path)
                                                break
                                            elif user_input.lower() == 'no':
                                                break
                                            else:
                                                print("Please enter 'yes' or 'no'.")
                                                continue
                                        merged_sim = utils.merge_sims(sim_txt_csvs,sim_log_csvs)
                                        merged_sim_sw = utils.join_sim_w_smartwatch(merged_sim, smartwatch_dfs)
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_sim_sw)
                                        continue
                                    else:
                                        print('Error selecting the files')
                                        continue

                                elif other_options == 'd':
                                    left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                                    sim_txt_files, sim_log_files = select_files_option_b('simulator')
                                    smartwatch = select_files_option_b('smartwatch')
                                    if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                                        gaze_left_csvs = load_dataframes(left_gaze_files)
                                        gaze_right_csvs = load_dataframes(right_gaze_files)
                                        gaze_ver_csvs = load_dataframes(ver_gaze_files)
                                        gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                                        gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                                        gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                                        gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                                        while True:
                                            user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                            if user_input.lower() == 'yes':
                                                print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                                sinc_path = file_path()
                                                gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                                break
                                            elif user_input.lower() == 'no':
                                                break
                                            else:
                                                print("Please enter 'yes' or 'no'.")
                                                continue
                                            
                                        sim_txt_csvs = load_dataframes(sim_txt_files)
                                        sim_log_csvs = load_dataframes(sim_log_files)  
                                        merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                                        merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample')  
                                        smartwatch_dfs = load_dataframes(smartwatch) 
                                        smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                                        if sinc_path: 
                                            smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch', sinc_path)
                                        
                                        merged_full = utils.join_smartwatch_txt(merged_sim_eyet, smartwatch_dfs, 'other')
                                        print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                        merge_files_or_folder(merged_full)
                                        continue
                                    else: 
                                        print('Error selecting the files')
                                        continue
                                    
                                elif other_options == 'e': 
                                    print('Exiting... Is it wrong? ')
                                    break
                                    
                                    
                                else:
                                    print('Choose a valid option - a, b, c, d or e')
                                    continue
                        elif join_option_b == 'f':
                            break
                            
                        else:
                            print('Choose a valid option - a, b, c or d')
                elif experiment == 'b': 
                    experiment == 'distraction'
                    join_option_b = input('Which ones do you want to join?\n a) Simulator Files\n b) Simulator + Eyetracker Files\n c) Simulator + Smartwatch Files\n d) Sim. + Smartwatch + Eyetracker Files\n e) Combinations with other files\n f) Go back\n>')
                    if join_option_b == 'a': 
                        sim_txt_files, sim_log_files = select_files_option_b('simulator')
                        if sim_txt_files and sim_log_files:
                        
                            sim_txt_csvs = load_dataframes(sim_txt_files)
                            sim_log_csvs = load_dataframes(sim_log_files)
                            merged_sim = utils.merge_sims(sim_txt_csvs, sim_log_csvs)
                            save_dfs_as_csv(merged_sim)
                            continue
                        else:
                            print("Error selecting the files.")
                            continue
                    elif join_option_b == 'b':
                        left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                        sim_txt_files, sim_log_files = select_files_option_b('simulator')
                        if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                            gaze_left_csvs = load_dataframes(left_gaze_files)
                            gaze_right_csvs = load_dataframes(right_gaze_files)
                            gaze_ver_csvs = load_dataframes(ver_gaze_files)
                            gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                            gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                            gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                            gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                            while True:
                                user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                if user_input.lower() == 'yes':
                                    print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                    sinc_path = file_path()
                                    gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                    break
                                elif user_input.lower() == 'no':
                                    break
                                else:
                                    print("Please enter 'yes' or 'no'.")
                                    continue     
                            sim_txt_csvs = load_dataframes(sim_txt_files)
                            sim_log_csvs = load_dataframes(sim_log_files) 
                            merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                            merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample') 
                            save_dfs_as_csv(merged_sim_eyet)    
                            return        
                        else:
                            print('Error selecting the files')
                            continue
                    elif join_option_b == 'c': 
                        sim_txt_files, sim_log_files = select_files_option_b('simulator')
                        print('\n Smartwatch File Selection:')
                        smartwatch = select_files_option_b('smartwatch')
                        if  sim_txt_files and sim_log_files:     
                            sim_txt_csvs = load_dataframes(sim_txt_files)
                            sim_log_csvs = load_dataframes(sim_log_files) 
                            smartwatch_dfs = load_dataframes(smartwatch) 
                            smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                            while True:
                                user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                if user_input.lower() == 'yes':
                                    print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                    sinc_path = file_path()
                                    smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch', sinc_path)
                                    break
                                elif user_input.lower() == 'no':
                                    break
                                else:
                                    print("Please enter 'yes' or 'no'.")
                                    continue
                            
                            merged_sim = utils.merge_sims(sim_txt_csvs,sim_log_csvs)
                            merged_sim_sw = utils.join_sim_w_smartwatch(merged_sim, smartwatch_dfs)
                            save_dfs_as_csv(merged_sim_sw)
                            return
                        else:
                            print('Error selecting the files')
                            continue
                    elif join_option_b == 'd':
                        left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                        sim_txt_files, sim_log_files = select_files_option_b('simulator')
                        smartwatch = select_files_option_b('smartwatch')
                        if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                            gaze_left_csvs = load_dataframes(left_gaze_files)
                            gaze_right_csvs = load_dataframes(right_gaze_files)
                            gaze_ver_csvs = load_dataframes(ver_gaze_files)
                            gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                            gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                            gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                            gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                            smartwatch_dfs = load_dataframes(smartwatch) 
                            smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                            while True:
                                user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                if user_input.lower() == 'yes':
                                    print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                    sinc_path = file_path()
                                    gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                    smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch', sinc_path)
                                    break
                                elif user_input.lower() == 'no':
                                    break
                                else:
                                    print("Please enter 'yes' or 'no'.")
                                    continue
                                
                            sim_txt_csvs = load_dataframes(sim_txt_files)
                            sim_log_csvs = load_dataframes(sim_log_files)  
                            merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                            merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample')  
                            merged_full = utils.join_smartwatch_txt(merged_sim_eyet, smartwatch_dfs, 'other')
                            save_dfs_as_csv(merged_full)
                            return
                        else: 
                            print('Error selecting the files')
                            continue
                    elif join_option_b == 'e':
                        print('Choose one of the following options:')
                        other_options = input('\n a) Simulator + Others\n b) Simulator + Eye Tracker + Others \n c) Simulador + Smartwatch + Others\n d) Simulator + Eyetracker + Smartwatch + Others\n e) Go back\n>')
                        if other_options == 'a':
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            if sim_txt_files and sim_log_files:
                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files)
                                merged_sim = utils.merge_sims(sim_txt_csvs, sim_log_csvs)
                                print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                merge_files_or_folder(merged_sim) 
                                continue
                            else:
                                print("Error selecting the files.")
                                
                                
                        elif other_options == 'b':
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                        
                            if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                                gaze_left_csvs = load_dataframes(left_gaze_files)
                                gaze_right_csvs = load_dataframes(right_gaze_files)
                                gaze_ver_csvs = load_dataframes(ver_gaze_files)
                                gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                                gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                                gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                                gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                                while True:
                                    user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                    if user_input.lower() == 'yes':
                                        print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                        sinc_path = file_path()
                                        gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                        break
                                    elif user_input.lower() == 'no':
                                        break
                                    else:
                                        print("Please enter 'yes' or 'no'.")
                                        continue
                                    
                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files) 
                                merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                                merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample')
                                print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible') 
                                merge_files_or_folder(merged_sim_eyet)  
                                continue          
                            else:
                                print('Error selecting the files')#estou aqui

                            
                        elif other_options == 'c':
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            print('\n Smartwatch File Selection:')
                            smartwatch = select_files_option_b('smartwatch')
                            if  sim_txt_files and sim_log_files:     
                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files) 
                                smartwatch_dfs = load_dataframes(smartwatch) 
                                smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                                while True:
                                    user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                    if user_input.lower() == 'yes':
                                        print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                        sinc_path = file_path()
                                        smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch', sinc_path)
                                        break
                                    elif user_input.lower() == 'no':
                                        break
                                    else:
                                        print("Please enter 'yes' or 'no'.")
                                        continue
                                
                                merged_sim = utils.merge_sims(sim_txt_csvs,sim_log_csvs)
                                merged_sim_sw = utils.join_sim_w_smartwatch(merged_sim, smartwatch_dfs)
                                print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                merge_files_or_folder(merged_sim_sw)
                                continue
                            else:
                                print('Error selecting the files')
                                continue

                        elif other_options == 'd':
                            left_gaze_files, right_gaze_files, ver_gaze_files  = select_files_option_b('eyetracker')
                            sim_txt_files, sim_log_files = select_files_option_b('simulator')
                            smartwatch = select_files_option_b('smartwatch')
                            if left_gaze_files and right_gaze_files and ver_gaze_files and sim_txt_files and sim_log_files:
                                gaze_left_csvs = load_dataframes(left_gaze_files)
                                gaze_right_csvs = load_dataframes(right_gaze_files)
                                gaze_ver_csvs = load_dataframes(ver_gaze_files)
                                gaze_left_csvs = prepare_eyetracker(gaze_left_csvs)
                                gaze_right_csvs = prepare_eyetracker(gaze_right_csvs)
                                gaze_ver_csvs = prepare_eyetracker(gaze_ver_csvs)
                                gaze = utils.join_vergences(gaze_left_csvs, gaze_right_csvs, gaze_ver_csvs)
                                smartwatch_dfs = load_dataframes(smartwatch) 
                                smartwatch_dfs = utils.prepare_smartwatch(smartwatch_dfs)
                                while True:
                                    user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                    if user_input.lower() == 'yes':
                                        print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                        sinc_path = file_path()
                                        gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                        smartwatch_dfs = utils.sinc_dataframes(smartwatch_dfs, 'smartwatch')
                                        break
                                    elif user_input.lower() == 'no':
                                        break
                                    else:
                                        print("Please enter 'yes' or 'no'.")
                                        continue
                                    
                                sim_txt_csvs = load_dataframes(sim_txt_files)
                                sim_log_csvs = load_dataframes(sim_log_files)  
                                merged_log_eyet = utils.join_log(gaze, sim_log_csvs)
                                merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_csvs, 'downsample')  
                                merged_full = utils.join_smartwatch_txt(merged_sim_eyet, smartwatch_dfs, 'other')
                                print('Select other files you may want to add to the dataset.\n DISCLAIMER: The files must have a column called "tempo(s)" so that the merge is possible')
                                merge_files_or_folder(merged_full)
                                continue
                            else: 
                                print('Error selecting the files')
                                continue
                            
                        elif other_options == 'e': 
                            print('Exiting...')
                            break
                        else:
                            print('Choose a valid option - a, b, c, d or e')
                            continue
                    elif join_option_b == 'f':
                        print('Going back...')
                        break

                    else:
                        print('Choose a valid option - a, b, c or d')
                        continue
                elif experiment == 'c':
                    print('Going back...')
                    break
        elif version == 'c': #versao c/ config files (features diferentes)
            print('You chose a option c. Your input file must be a .csv file and it needs the separator must be ";".')
            ####################
            experiment = input('a) Drowsiness\nb) Distraction\nc) Go back\n>')    
            if experiment == 'a':
                sim_txt_list = []
                sim_log_list = []
                gaze_left_list = []
                gaze_right_list = []
                gaze_ver_list = []
                sim_txt = None
                sim_log= None
                smartwatch_list = []
                experiment = 'drowsiness'
                print('Escolha o formato do ficheiro que pretende extrair :')
                #while True:
                device = input('a) Eyetracker - Vergences\nb) Simulator Files\nc) Smartwatch\nd) Go back\n>')
                if device == 'a':
                    config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/config_eyetracker.json'
                    #while True:
                    gaze_type = input('Do you wish to extract which gaze type?\na) Left\nb) Right\nc) Vergence\nd) Go Back\n>')
                    if gaze_type in ['a', 'b', 'c', 'd']:
                        if gaze_type == 'a':
                            gaze_left = config(config_file)
                            gaze_left_list.append(gaze_left)
                            break
                            
                        elif gaze_type == 'b':
                            gaze_right = config(config_file)
                            gaze_right_list.append(gaze_right)
                            break
                        elif gaze_type == 'c':
                            gaze_ver = config(config_file)
                            gaze_ver_list.append(gaze_ver)
                            break
                        elif gaze_type == 'd':
                            break
                        else:
                            print('Choose a correct option - a, b or c')
                elif device == 'b':
                    #while True:
                    simulator_type = input('Do you wish to extract which simulator file?\na) Car Data\nb) Participant Data (KSS, etc.)\nc) Go back\n>')
                    if simulator_type in ['a', 'b', 'c']:
                        if simulator_type == 'a':
                            config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/config_simtxt.json'
                            sim_txt = config(config_file)
                            sim_txt_list.append(sim_txt)
                            continue
                        elif simulator_type == 'b':
                            config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/config_log.json'
                            sim_log = config(config_file)
                            sim_log_list.append(sim_log)
                        elif simulator_type == 'c':
                            continue
                
                        else:
                            print('Choose a correct option - a, b or c')
                            continue
                elif device == 'c':
                    config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/config_smartwatch.json'
                    smartwatch = config(config_file)
                    smartwatch_list.append(smartwatch)
                    continue
                elif device == 'd':
                    break

                else:
                    print('Choose a valid option - a, b, c or d')
                    
                

            elif experiment == 'b':
                experiment = 'distraction'
                sim_txt_list = []
                sim_log_list = []
                gaze_left_list = []
                gaze_right_list = []
                gaze_ver_list = []
                sim_txt = None
                sim_log= None
                smartwatch_list = []
                print('Escolha o formato do ficheiro que pretende extrair :')
                #while True:
                device = input('a) Eyetracker - Vergences\nb) Simulator Files\nc) Smartwatch\nd) Go back\n>')
                if device == 'a':
                    config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/config_eyetracker.json'
                    while True:
                        gaze_type = input('Do you wish to extract which gaze type?\na) Left\nb) Right\nc) Vergence\n>')
                        if gaze_type in ['a', 'b', 'c']:
                            if gaze_type == 'a':
                                gaze_left = config(config_file)
                                gaze_left_list.append(gaze_left)
                            elif gaze_type == 'b':
                                gaze_right = config(config_file)
                                gaze_right_list.append(gaze_right)
                            else:
                                gaze_ver = config(config_file)
                                gaze_ver_list.append(gaze_ver)
                            break
                        else:
                            print('Choose a correct option - a, b or c')
                elif device == 'b':
                    # while True:
                    simulator_type = input('Do you wish to extract which simulator file?\na) Car Data\nb) Participant Data (KSS, etc.)\n>')
                    if simulator_type in ['a', 'b']:
                        if simulator_type == 'a':
                            config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/config_simtxt.json'
                            sim_txt = config(config_file)
                            sim_txt_list.append(sim_txt)
                        else:
                            config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/dist_config_log.json'
                            sim_log = config(config_file)
                            sim_log_list.append(sim_log)
                        break
                    else:
                        print('Choose a correct option - a or b')
                elif device == 'c':
                    config_file = '/Users/henriqueribeiro/Desktop/Tese/Configuration Files/config_smartwatch.json'
                    smartwatch = config(config_file)
                    smartwatch_list.append(smartwatch)
                elif device == 'd':
                    continue

                else:
                    print('Choose a valid option - a, b, c or d')
                    continue
                    
                next_step = input('You have now extracted the files. What would you like to do next?\n1) Configure another file\n2) Join files\n3) Exit\n>')
                if next_step == '3':
                    print("Exiting...")
                    break
                elif next_step == '2':# Before using sim_txt and sim_log
                    join_option = input('You chose to join the newly mapped/extracted files. Which ones do you want to join?\n a) Simulator Files\n b) Simulator + Eyetracker Files\n c) Simulator + Smartwatch Files\n d) Sim. + Smartwatch + Eyetracker Files\n e) Go back\n>')
                    if join_option == 'a': 
                        merged_sim = utils.merge_sims(sim_txt_list, sim_log_list)
                    elif join_option == 'b':
                        gaze = utils.join_vergences(gaze_left_list, gaze_right_list, gaze_ver_list)
                        while True:
                                user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                                if user_input.lower() == 'yes':
                                    print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                    sinc_path = file_path()
                                    gaze = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                    break
                                elif user_input.lower() == 'no':
                                    break
                                else:
                                    print("Please enter 'yes' or 'no'.")
                                    continue
                        merged_log_eyet = utils.join_log(gaze, sim_log_list)
                        merged_sim_eyet = utils.join_smartwatch_txt(merged_log_eyet, sim_txt_list, 'downsample')
                    elif join_option == 'c':
                        merged_sims = utils.merge_sims(sim_txt_list, sim_log_list)
                        merged_sim_sw = utils.join_sim_w_smartwatch(merged_sims, smartwatch_list)
                    elif join_option == 'd': 
                        gaze_d = utils.join_vergences(gaze_left_list, gaze_right_list, gaze_ver_list)
                        while True:
                            user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
                            if user_input.lower() == 'yes':
                                print('Select the file containing the .csv with the data to synchronize the dataframes.')
                                sinc_path = file_path()
                                gaze_d = utils.sinc_dataframes(gaze, 'eyetracker', sinc_path)
                                break
                            elif user_input.lower() == 'no':
                                break
                            else:
                                print("Please enter 'yes' or 'no'.")
                                continue
                        merged_df = utils.join_log(gaze_d, sim_log_list)
                        merged = utils.join_smartwatch_txt(merged_df, sim_txt_list, 'downsample')
                        merged_full = utils.join_smartwatch_txt(merged, smartwatch_list, 'other')
                    elif join_option == 'e':
                        continue
                    else:
                        print('Choose a valid option - a, b, c, d or e')
                    
                    pass

            
                    # No need to explicitly handle '1' as the loop will continue automatically
            elif experiment == 'c':
                continue
            else:
                print('Choose a valid experiment type - a, b or c')
        elif version == 'd':
            print('Exiting...')
            break  
        else: 
            print('Escolha uma opção válida - a, b, c ou d')

if __name__ == "__main__":
    main()
