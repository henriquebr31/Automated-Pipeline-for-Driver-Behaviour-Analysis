import os
import sys
import re
import time
import datetime
import math
import pandas as pd
import glob
import numpy as np
import math
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import OneHotEncoder
import pyreadstat
from tkinter import filedialog, messagebox
import configuration
import json



def log2str(logfile):
    if not os.path.exists(logfile):
        print("File does not exist!")
        return
    with open(logfile, "r") as f:
        lines = f.readlines()
    return "".join(lines)


def filter_dataframes_by_features(dfs):
    # Load the list of features from the JSON file
    json_path = configuration.file_path()
    with open(json_path[0], 'r') as file:
        features = json.load(file)
    
    filtered_dfs = []
    
    for df in dfs:
        # Identify features that are not in the DataFrame
        missing_features = [feature for feature in features if feature not in df.columns]
        
        if missing_features:
            # Check if all features are missing
            if len(missing_features) == len(features):
                print('No features in the JSON are present in the original DataFrame. Returning the original one...')
                filtered_dfs.append(df)
            else:
                print(f"Features {', '.join(missing_features)} not in the original DataFrame.")
                # Filter the DataFrame to include only the present features
                present_features = [feature for feature in features if feature in df.columns]
                filtered_dfs.append(df[present_features])
        else:
            # If no features are missing, filter the DataFrame as usual
            filtered_dfs.append(df[features])
    
    return filtered_dfs


def extract_features(telnav, experiment): 
    timestamp_array = []
    state_array = []
    info_array = []
    previous_timestamp = None
    for i in range(len(telnav)):
        timestamp = telnav[i].split(" ")[0]
        #debugging the timestamp
        timestamp = timestamp.split(',')[0]
        pattern = '0{2,}'
        match = re.search(pattern, timestamp[::-1])
        if match:
            # Calculate the original indices of the match
            end_index = len(timestamp) - match.start()
            start_index = len(timestamp) - match.end()

            # Extract the parts of the string before and after the matched pattern
            string_1 = timestamp[:start_index]
            string_2 = timestamp[start_index:]
        else:
            # If no match is found, the original string does not contain two or more consecutive zeros
            string_1 = timestamp
            string_2 = ''
        timestamp_debugged = string_2 + string_1 

        timestamp_debugged = int(timestamp_debugged) #this is still not 100% correct - function fix_timestamps deals with it :) 
        timestamp_array.append(timestamp_debugged)
        state = telnav[i].split(" ")[3]
        state_array.append(state)
        if experiment == 'drowsiness':
            if len(telnav[i].split(" ")) == 7 and telnav[i].split(" ")[6] == "Timer":           
                info_array.append("Periodic Timer")
            elif len(telnav[i].split(" ")) == 7 and telnav[i].split(" ")[6] != "Timer":     
                info = telnav[i].split(" ")[6]
                info_array.append(info)
            else:
                info_array.append("Timer")
        elif experiment == 'distraction':
            command = telnav[i].split("-")[1]
            info_array.append(command)
        
        
    return timestamp_array, state_array, info_array


def fix_timestamps(df):
    df['Timestamp'] = (df['Timestamp'] / 60000000) * 60
    df['Timestamp'] = pd.to_numeric(df['Timestamp'])

    if len(df) > 1 and df.loc[0, 'Timestamp'] * 10 < df.loc[1, 'Timestamp']:
        df.loc[0, 'Timestamp'] *= 10

    # Initialize a variable to hold the value of the last timestamp
    last_timestamp = df.loc[0, 'Timestamp']

    for i in range(1, len(df)):
        current_timestamp = df.loc[i, 'Timestamp']
        if current_timestamp < last_timestamp:
            if current_timestamp * 100 <= last_timestamp:
                current_timestamp *= 1000
                df.loc[i, 'Timestamp'] = current_timestamp
            elif current_timestamp * 10 < last_timestamp:
                current_timestamp *= 100
                df.loc[i, 'Timestamp'] = current_timestamp
            
            else: 
                current_timestamp *= 10
                df.loc[i, 'Timestamp'] = current_timestamp


        # Update the last_timestamp to the newly adjusted current_timestamp
        last_timestamp = current_timestamp

    return df

################################################ REFINED MV IMPUTATION ################################################


def count_initial_null_rows(df):

    count = 0
    for index, row in df.iterrows():
        if pd.isna(row['KSS']):  # Check if all values in the row are null
            count += 1
        else:
            break  # Stop counting when a row with at least one non-null value is found
    return count

def impute_first(df): 
    df['Fiabilidade'] = 1
    count = count_initial_null_rows(df)
    if count <=2 and count > 0:
        df.loc[:count, 'KSS'] = df.loc[count , 'KSS']
        df.loc[:count-1, 'Fiabilidade'] = 0
    else: 
        df = df[count:]
    return df 

def find_null_sequences(df):
    null_sequences = []
    null_sequence_length = 0
    for idx, value in df.iterrows():
        if pd.isnull(value['KSS']):
            null_sequence_length += 1
            if null_sequence_length == 1:
                start_idx = idx
            if null_sequence_length >= 2:
                continue
        else:
            if null_sequence_length >= 2:
                null_sequences.append((start_idx, idx - 1))
            null_sequence_length = 0
    if null_sequence_length >= 2:
        null_sequences.append((start_idx, idx))
    return null_sequences

def slice_lengths(df):
    sequences = find_null_sequences(df)
    list_of_dfs = []
    if sequences:
        for i in range(len(sequences)+1):
            df_name = f'df_{i}'
            if i == 0:
                #print("Length of sequence before:", len(df[:sequences[i][0]]))
                df_name = df[:sequences[i][0]]
                list_of_dfs.append(df_name)
            elif i == len(sequences):
                #print('Length of sequence after:', len(df[sequences[i-1][1] + 1:]))
                df_name = df[sequences[i-1][1] + 1:]
                list_of_dfs.append(df_name)
            elif i:
                #print(f'Length of sequence between for i = {i}:', len(df[sequences[i-1][1] + 1:sequences[i][0]]))
                df_name = df[sequences[i-1][1] + 1:sequences[i][0]]
                list_of_dfs.append(df_name)
        df_with_highest_length = max(list_of_dfs, key=lambda df: len(df))
        return df_with_highest_length
    else:
        return df


def impute_mean(df):
    for i in range(1, len(df) - 1):
        if pd.isnull(df.iloc[i]['KSS']):
            df.iloc[i, df.columns.get_loc('KSS')] = math.ceil((df.iloc[i - 1]['KSS'] + df.iloc[i + 1]['KSS']) / 2)
            #put fiabilidade to 0 in the row of the imputed value
            df.iloc[i, df.columns.get_loc('Fiabilidade')] = 0
    return df

def mv(df):
    first_none_index = None
    df = impute_first(df)
    

    for index, value in enumerate(df['KSS']):
        if pd.isna(value):
            if not df['KSS'][index+1:].notna().any():
                
                first_none_index = index
                break
    if first_none_index is not None:
        if df.loc[first_none_index - 1, 'KSS'] > 5:
           
            df.loc[first_none_index:, 'KSS'] = df.loc[first_none_index:, 'KSS'].fillna(10)

            df.loc[first_none_index:, 'Fiabilidade'] = 0
        elif df.loc[first_none_index - 1, 'KSS'] <= 5:
            
            df.loc[first_none_index:, 'KSS'] = df.loc[first_none_index:, 'KSS'].fillna(0)
            

            df.loc[first_none_index:, 'Fiabilidade'] = 0
    

    df = df.reset_index(drop=True)
    df = slice_lengths(df)
    df = impute_mean(df)


    return df
###########################################GAZE MV IMPUTATION##########################################################

def find_big_null(df):
    null_sequences = []
    null_sequence_length = 0
    for idx, value in df.iterrows():
        if pd.isnull(value['X Pos']):
            null_sequence_length += 1
            if null_sequence_length == 1:
                start_idx = idx
            if null_sequence_length > 2:  # Changed from `>= 1` to `>= 2`
                continue
        else:
            if null_sequence_length > 2:
                null_sequences.append((start_idx, idx - 1))
            null_sequence_length = 0
    if null_sequence_length > 2:
        null_sequences.append((start_idx, idx))
    return null_sequences


def find_small_null(df):
    null_sequences = []
    null_sequence_length = 0
    start_idx = 0

    for idx in range(len(df)):
        if pd.isnull(df.loc[idx, 'X Pos']):
            null_sequence_length += 1
            if null_sequence_length == 1:
                start_idx = idx
            elif null_sequence_length == 2:
                if idx == len(df) - 1 or not pd.isnull(df.loc[idx + 1, 'X Pos']):
                    null_sequences.append((start_idx, idx))
        else:
            if null_sequence_length == 1:
                null_sequences.append((start_idx, idx - 1))
            null_sequence_length = 0

    if null_sequence_length == 1:
        null_sequences.append((start_idx, len(df) - 1))

    return null_sequences

def flatten_ranges(list_of_ranges):
    flattened_list = []
    for start, end in list_of_ranges:
        flattened_list.extend(range(start, end + 1))
    return flattened_list

def impute_gaze_mv(df):
    # Find big nulls and fill with 0
    big_nulls = find_big_null(df)
    for start, end in big_nulls:
        df.loc[start:end, ['X Pos', 'Y Pos', 'Pupil Diameter']] = 0

    # Find small nulls and interpolate
    small_nulls = find_small_null(df)
    for col in ['X Pos', 'Y Pos', 'Pupil Diameter']:
        for start, end in small_nulls:
            prev_idx = max(start - 1, 0)
            next_idx = min(end + 1, len(df) - 1)
            prev_value = df.loc[prev_idx, col]
            next_value = df.loc[next_idx, col]
            df.loc[start:end, col] = np.nanmean([prev_value, next_value])

    return df



########################################################Feature Extraction#############################################

# def  outlier_perclos(df, criterion):

#     Q1 = df['Pupil Diameter'].quantile(0.25)
#     Q3 = df['Pupil Diameter'].quantile(0.75)
#     IQR = Q3 - Q1

#     k = 3
#     upper_threshold = Q3 + k * IQR

#     is_outlier = df['Pupil Diameter'] > upper_threshold

   
#     if criterion == 'remove':
#     # remove rows containing outliers
#         df_clean = df[df['Pupil Diameter'] <= upper_threshold]
#         dataframe = df_clean.copy()
#         max_value = dataframe['Pupil Diameter'].max()
#         min_value = dataframe['Pupil Diameter'].min()

        
        
#         # Calculate perclos for each row
#         dataframe.loc[:,'perclos'] = ((dataframe['Pupil Diameter'] - min_value) / (max_value - min_value)) * 100

#         return dataframe
#     elif criterion == 'replace':
#         dataframe = df.copy()
#         max_value = dataframe.loc[~is_outlier, 'Pupil Diameter'].max()
#         min_value = dataframe.loc[~is_outlier, 'Pupil Diameter'].min()

#         # Calculate perclos for each row except for the outliers
#         dataframe.loc[~is_outlier, 'perclos'] = ((dataframe.loc[~is_outlier, 'Pupil Diameter'] - min_value) / (max_value - min_value)) * 100
#         # Set perclos value '101' for outliers
#         dataframe.loc[is_outlier, 'perclos'] = 101

#         return dataframe
#     else:
#         raise ValueError("Criterion must be 'remove' or 'replace'.")

def  outlier_perclos(df, criterion):

    Q1 = df['Pupil Diameter'].quantile(0.25)
    Q3 = df['Pupil Diameter'].quantile(0.75)
    IQR = Q3 - Q1

    k = 3
    upper_threshold = Q3 + k * IQR

    is_outlier = df['Pupil Diameter'] > upper_threshold

   
    if criterion == 'remove':
    # remove rows containing outliers
        df_clean = df[df['Pupil Diameter'] <= upper_threshold]
        dataframe = df_clean.copy()
        max_value = dataframe['Pupil Diameter'].max()
        min_value = dataframe['Pupil Diameter'].min()

        
        
        # Calculate perclos for each row
        dataframe.loc[:,'pupil_percentage'] = ((dataframe['Pupil Diameter'] - min_value) / (max_value - min_value)) * 100

        return dataframe
    elif criterion == 'replace':
        dataframe = df.copy()
        max_value = dataframe.loc[~is_outlier, 'Pupil Diameter'].max()
        min_value = dataframe.loc[~is_outlier, 'Pupil Diameter'].min()

        # Calculate perclos for each row except for the outliers
        dataframe.loc[~is_outlier, 'pupil_percentage'] = ((dataframe.loc[~is_outlier, 'Pupil Diameter'] - min_value) / (max_value - min_value)) * 100
        # Set perclos value '101' for outliers
        dataframe.loc[is_outlier, 'pupil_percentage'] = 999 
        # Set perclos value '100' for null values
        dataframe['pupil_percentage'] = dataframe['pupil_percentage'].fillna(100) #porquê 100? - rever

        return dataframe
    else:
        raise ValueError("Criterion must be 'remove' or 'replace'.")
    

def create_open_eye(df):
    df['eye open'] = np.where(df['pupil_percentage'] > 25, 1, 0)
    return df

def real_perclos(df):
    # Define the interval duration
    interval_duration = 60
    
    # Calculate the interval start for each timestamp
    df['interval_start'] = (df['Start Time (secs)'] // interval_duration) * interval_duration
    
    # Initialize a DataFrame to store the results
    percentage_df = df[['interval_start', 'eye open']].groupby('interval_start').apply(
        lambda x: (x['eye open'].mean() * 100)
    ).reset_index(name='perclos')
    
    # Merge the percentages back to the original DataFrame
    df = df.merge(percentage_df, on='interval_start', how='left')
    
    # Drop the 'interval_start' column as it's no longer needed
    df = df.drop(columns=['interval_start'])
    
    return df
     


#######################################################################################################################


def impute_mv_log(df):
    for i in range(len(df)):
        if pd.isnull(df.iloc[i]['KSS']):
            if i == 0:
                next_non_null_index = i + 1
                while next_non_null_index < len(df) and pd.isnull(df.iloc[next_non_null_index]['KSS']):
                    next_non_null_index += 1
                
                if next_non_null_index < len(df):
                    df.iloc[i, df.columns.get_loc('KSS')] = df.iloc[next_non_null_index]['KSS']
                else:
                    df.iloc[i, df.columns.get_loc('KSS')] = df.iloc[-1]['KSS']
            elif i == len(df) - 1:
                prev_non_null_index = i - 1
                while prev_non_null_index >= 0 and pd.isnull(df.iloc[prev_non_null_index]['KSS']):
                    prev_non_null_index -= 1
                
                if prev_non_null_index >= 0:
                    df.iloc[i, df.columns.get_loc('KSS')] = df.iloc[prev_non_null_index]['KSS']
                else:
                    df.iloc[i, df.columns.get_loc('KSS')] = df.iloc[0]['KSS']
            else:
                next_non_null_index = i + 1
                while next_non_null_index < len(df) and pd.isnull(df.iloc[next_non_null_index]['KSS']):
                    next_non_null_index += 1
                
                prev_non_null_index = i - 1
                while prev_non_null_index >= 0 and pd.isnull(df.iloc[prev_non_null_index]['KSS']):
                    prev_non_null_index -= 1
                
                if next_non_null_index < len(df) and prev_non_null_index >= 0:
                    df.iloc[i, df.columns.get_loc('KSS')] = (df.iloc[prev_non_null_index]['KSS'] + df.iloc[next_non_null_index]['KSS']) / 2
                elif next_non_null_index < len(df):
                    df.iloc[i, df.columns.get_loc('KSS')] = df.iloc[next_non_null_index]['KSS']
                elif prev_non_null_index >= 0:
                    df.iloc[i, df.columns.get_loc('KSS')] = df.iloc[prev_non_null_index]['KSS']
                else:
                    df.iloc[i, df.columns.get_loc('KSS')] = 0  # Fallback value if all surrounding values are null
    return df

  

def create_df(file_path, experiment, j):
    telnav = log2str(file_path)
    telnav = telnav.split("\n")
    #save first and last lines
    data = telnav[0] + "\n" + telnav[1] + "\n" + telnav[2] + "\n"
    last = telnav[-1]
    #delete first three lines and the last ones from telnav
    telnav = telnav[3:]
    telnav = telnav[:-2]
    timestamp_array, _ , info_array = extract_features(telnav, experiment)
    df = pd.DataFrame(list(zip(timestamp_array, info_array)), columns =['Timestamp', 'KSS'])
    df['ID'] = j.split('_')[1]
    cols = ['ID'] + [col for col in df if col != 'ID']
    df_raw = df[cols]

    # Check for lines with 'Timer' following 'Periodic Timer' and insert null values
    insert_indices = []
    for i, info in enumerate(df_raw['KSS']):
        if 'Periodic Timer' in info and i + 1 < len(df_raw) and 'Timer' in df_raw.iloc[i + 1]['KSS']:
            insert_indices.append(i + 1)


    for index in reversed(insert_indices):
        # Calculate the mean timestamp for the new row
        prev_timestamp = df_raw.iloc[index - 1]['Timestamp']
        next_timestamp = df_raw.iloc[index]['Timestamp']
        mean_timestamp = int((prev_timestamp + next_timestamp) / 2)

        # Use ID from the previous row
        id_value = df_raw.iloc[index - 1]['ID']
        
        # Insert the new row with specified values
        new_row = pd.DataFrame({
            'ID': [id_value],  # Use the same ID as the last row
            'Timestamp': [mean_timestamp],  # Mean of the next and previous row's timestamps
            'KSS': [None]  # KSS is a null value
        }, index=[index + 0.5])  # Use a fractional index for easier concatenation
        
        df_raw = pd.concat([df_raw.iloc[:index], new_row, df_raw.iloc[index:]]).reset_index(drop=True)

    df_raw = df_raw[~df_raw['KSS'].str.contains('timer|periodic timer', case=False, na=False)]
    #convert KSS to int
    df_raw['KSS'] = df_raw['KSS'].astype(float)
    
    df_clean = df_raw
    
    #reset index of the dataframe
    df_clean = df_clean.reset_index(drop=True)

    df_clean = mv(df_clean)
    

    return df_clean

def calculate_timestamps(ts): 
    ts = ts.split(',')[0]
    pattern = '0{2,3}' #mudar aqui
    match = re.search(pattern, ts[::-1])
    if match:
        # Calculate the original indices of the match
        end_index = len(ts) - match.start()
        start_index = len(ts) - match.end()
        string_1 = ts[:start_index]
        string_2 = ts[start_index:]
    else:
        # If no match is found, the original string does not contain two or more consecutive zeros
        string_1 = ts
        string_2 = ''
    timestamp_debugged = string_2 + string_1 

    return int(timestamp_debugged)

def distraction_log(file_path):
    new_df = pd.DataFrame(columns=['Timestamp', 'Event'])
    rows = []
    telnav = log2str(file_path)
    telnav_clean = telnav.split('\n')[3:][:-2]
    

    for line in telnav_clean:
        time = line.split(' ')[0]
        event = line.split(' ')[3]
        rows.append({'Timestamp': time, 'Event': event})

    # Use pd.concat to create the DataFrame from the list of dictionaries
    new_df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)
    new_df['Timestamp'] = new_df['Timestamp'].apply(calculate_timestamps)
    df_final = fix_timestamps(new_df)

    return df_final

    


# def sinc_dataframes(to_sinc, device):

#     sinc_csv = pd.read_csv('/Users/henriqueribeiro/Desktop/Tese/sincronizacao_prof_rute/data_horas_sono.csv', sep=';')
#     sinc_csv = sinc_csv[['participant','EyeTracking_start_time_corretion_secs', 'Smartwatch_start_time_corretion_secs']]
#     #delete rows with NaN values
#     sinc_csv = sinc_csv.dropna()
    
#     #copy to_sinc to avoid modifying the original dataframe
#     to_sinc_copy = to_sinc.copy()

#     if device == 'smartwatch':
#         for df in to_sinc_copy: 
#             participant = df['Participant'].iloc[0]
#             #find the corresponding row in the sinc_csv dataframe
#             sinc_row = sinc_csv[sinc_csv['participant'] == int(participant)]
#             if not sinc_row.empty:
#                 sinc_time = sinc_row['Smartwatch_start_time_corretion_secs'].iloc[0]
#                 #subtract the sinc_time from every row of the timestamp column
#                 df['secs'] = df['secs'] + sinc_time
#                 #df['tempo(s)'] = df['tempo(s)'] + sinc_time
#             # else: 
#             #     print(f'The participant {participant} is not in the sinc_csv dataframe')
#     elif device == 'eyetracker':
#         for df in to_sinc_copy:
#             participant = df['Participant'].iloc[0]
#             #find the corresponding row in the sinc_csv dataframe
#             sinc_row = sinc_csv[sinc_csv['participant'] == int(participant)]
#             if not sinc_row.empty:
#                 sinc_time = sinc_row['EyeTracking_start_time_corretion_secs'].iloc[0]
#                 #subtract the sinc_time from every row of the timestamp column
#                 df['Start Time (secs)'] = df['Start Time (secs)'] + sinc_time
#             # else: 
#             #     print(f'The participant {participant} is not in the sinc_csv dataframe')
#     return to_sinc

def sinc_dataframes(to_sinc, device, base_path):

    sinc_csv = pd.read_csv(base_path, sep=';')
    sinc_csv = sinc_csv[['participant','EyeTracking_start_time_corretion_secs', 'Smartwatch_start_time_corretion_secs']]
    #delete rows with NaN values
    sinc_csv = sinc_csv.dropna()
    
    #copy to_sinc to avoid modifying the original dataframe
    to_sinc_copy = to_sinc.copy()

    to_sinc_copy = [df for df in to_sinc if isinstance(df, pd.DataFrame) and 'Participant' in df.columns]


    if device == 'smartwatch':
        for df in to_sinc_copy: 
            # if df == '.DS_Store':
            #     continue
            participant = df['Participant'].iloc[0]
            #find the corresponding row in the sinc_csv dataframe
            sinc_row = sinc_csv[sinc_csv['participant'] == int(participant)]
            if not sinc_row.empty:
                sinc_time = sinc_row['Smartwatch_start_time_corretion_secs'].iloc[0]
                #subtract the sinc_time from every row of the timestamp column
                df['secs'] = df['secs'] + sinc_time
                #df['tempo(s)'] = df['tempo(s)'] + sinc_time
            # else: 
            #     print(f'The participant {participant} is not in the sinc_csv dataframe')
    elif device == 'eyetracker':
        for df in to_sinc_copy:
            # if df == '.DS_Store':
            #     continue
            participant = df['Participant'].iloc[0]
            #find the corresponding row in the sinc_csv dataframe
            sinc_row = sinc_csv[sinc_csv['participant'] == int(participant)]
            if not sinc_row.empty:
                sinc_time = sinc_row['EyeTracking_start_time_corretion_secs'].iloc[0]
                #subtract the sinc_time from every row of the timestamp column
                df['Start Time (secs)'] = df['Start Time (secs)'] + sinc_time
            # else: 
            #     print(f'The participant {participant} is not in the sinc_csv dataframe')
    return to_sinc

def create_simulator_df(file_path):
    with open(file_path, 'r') as file:
    # Read the entire file content into a string
        simulator = file.read()
    simulator_init = simulator.split("\n")

    dados1 = simulator_init[6:10][0].split(':')  
    comprimento = dados1[2].split(' ')[0]
    largura = dados1[3].split(' ')[0]
    eixos = dados1[4]
    dados2 = simulator_init[6:10][1]
    posicao_frente = dados2.split(';')[0].split(':')[1].split('m')[0]
    posicao_direita = dados2.split(';')[1].split('m')[0]
    dados3 = simulator_init[6:10][3]
    guia_longitudinal = dados3.split(':') [1].split('m')[0]  

    #generate empty dictionary
    simulator_dict = {}
    #fill dictionary with data
    simulator_dict['comprimento'] = float(comprimento)
    simulator_dict['largura'] = float(largura)
    simulator_dict['eixos'] = float(eixos)
    simulator_dict['posicao_frente'] = float(posicao_frente)
    simulator_dict['posicao_direita'] = float(posicao_direita)
    simulator_dict['guia_longitudinal'] = float(guia_longitudinal)
    #delete first 10 lines of simulator string
    simulator = simulator_init[10:]
    #delete last line of simulator string
    simulator = simulator[:-2]
    header = simulator[1].split("\t")
    # Convert all rows except the first two (or more if needed) into a list of lists
    data = [row.split("\t") for row in simulator[2:]]

    # Create the DataFrame
    simulator_df = pd.DataFrame(data, columns=header)

    return simulator_df, simulator_dict 

def select_base_path():
    # Open the dialog to choose a folder, this will freeze the script execution until a folder is selected
    # Create a root window but hide it
    print('Select the main files:')
    while True:
        root = tk.Tk()
        root.withdraw()
        base_path = filedialog.askdirectory()

        if base_path and os.path.isdir(base_path):
                return base_path
        else:
            messagebox.showerror(" Invalid Directory","You clicked cancel.\n Exiting...")
            print('Exiting...')
            sys.exit()
            

    
    

def extract_all(experiment, data_source, eyetracker_type,  file_type, base_path):
    #create a empty dataframe
    list_df = []
    while True: 
        try:
        ############################## DISTRACTION CSV ########################################
            if experiment == 'distraction' and file_type == 'csv':
                #base_path = '/Users/henriqueribeiro/Desktop/tese/data/Experimentos BBAI/distracao extraido'
                #prompt the user for additional information
                # print('Please provide the data source (e.g. "smartwatch"):')
                # data_source = input()
                #print('The data source chosen was: ' + data_source + '\n')
                if data_source == 'smartwatch': 
                    for i in os.listdir(base_path):  
                        folder = f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_smartwatch_*_*_*/*.csv'
                        matches = glob.glob(folder)
                        # Check if there is at least one match
                        if matches:
                            df = pd.read_csv(matches[0])
                            df['Participant'] = i.split('_')[1]
                            #append the dataframe to the empty dataframe
                            list_df.append(df)
                            print('Dataframe created!')            
                if data_source == 'eyetracker':
                    # print('Blink, Gaze or Fixation? Left, Right or Vergence? (e.g. "blink left"):')
                    # eyetracker_type = input()
                    #blink
                    if eyetracker_type == 'blink left':
                        for i in os.listdir('/Users/henriqueribeiro/Desktop/tese/data/Experimentos BBAI/distracao extraido'):   
                            folder = f'/Users/henriqueribeiro/Desktop/tese/data/Experimentos BBAI/distracao extraido/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*_Default_60sec Blink-Left.csv'
                            matches = glob.glob(folder)
                        
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                        print('Dataframe created!')            
                    if eyetracker_type == 'blink right':
                        for i in os.listdir('/Users/henriqueribeiro/Desktop/tese/data/Experimentos BBAI/distracao extraido'):   
                            folder = f'/Users/henriqueribeiro/Desktop/tese/data/Experimentos BBAI/distracao extraido/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*_Default_60sec Blink-Right.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                        print('Dataframe created!')            
                    if eyetracker_type == 'blink vergence':
                        for i in os.listdir('/Users/henriqueribeiro/Desktop/tese/data/Experimentos BBAI/distracao extraido'):   
                            folder = f'/Users/henriqueribeiro/Desktop/tese/data/Experimentos BBAI/distracao extraido/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*_Default_60sec Blink-Vergence.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                        print('Dataframe created! ')              
                    #gaze
                    if eyetracker_type == 'gaze left':
                        for i in tqdm(os.listdir(base_path)):
                            # Define the patterns to search for
                            patterns = [
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/*/* Gaze-Left.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/* Gaze-Left.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Left.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/* Gaze-Left.csv'
                            ]
                            
                            # Initialize variable to keep track of whether a match was found
                            found_match = False
                            
                            # Iterate through patterns until a match is found
                            for pattern in patterns:
                                matches = glob.glob(pattern)
                                if matches:
                                    df = pd.read_csv(matches[0], delimiter=';')
                                    df['Participant'] = i.split('_')[1]
                                    df = impute_gaze_mv(df)
                                    df = outlier_perclos(df,'replace')
                                    df = create_open_eye(df)
                                    df = real_perclos(df)
                                    list_df.append(df)
                                    found_match = True
                                    break  # Exit the loop once the first match is found
                            
                            # if not found_match:
                            #     print(f'This one is not being counted: {i}')       
                    if eyetracker_type == 'gaze right':
                        for i in tqdm(os.listdir(base_path)):
                            # Define the patterns to search for
                            patterns = [
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/*/* Gaze-Right.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/* Gaze-Right.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Right.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/* Gaze-Right.csv'
                            ]
                            
                            # Initialize variable to keep track of whether a match was found
                            found_match = False
                            
                            # Iterate through patterns until a match is found
                            for pattern in patterns:
                                matches = glob.glob(pattern)
                                if matches:
                                    df = pd.read_csv(matches[0], delimiter=';')
                                    df['Participant'] = i.split('_')[1]
                                    df = impute_gaze_mv(df)
                                    df = outlier_perclos(df,'replace')
                                    df = create_open_eye(df)
                                    df = real_perclos(df)
                                    list_df.append(df)
                                    found_match = True
                                    break  # Exit the loop once the first match is found
                            
                            if not found_match:
                                print(f'This one is not being counted: {i}')             
                    if eyetracker_type == 'gaze vergence':
                        for i in tqdm(os.listdir(base_path)):
                            # Define the patterns to search for
                            patterns = [
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/*/* Gaze-Vergence.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/* Gaze-Vergence.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Vergence.csv',
                                f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/* Gaze-Vergence.csv'
                            ]            
                            # Initialize variable to keep track of whether a match was found
                            found_match = False       
                            # Iterate through patterns until a match is found
                            for pattern in patterns:
                                matches = glob.glob(pattern)
                                if matches:
                                    df = pd.read_csv(matches[0], delimiter=';')
                                    df['Participant'] = i.split('_')[1]
                                    df = impute_gaze_mv(df)
                                    list_df.append(df)
                                    found_match = True
                                    break  # Exit the loop once the first match is found
                            
                            # if not found_match:
                            #     print(f'This one is not being counted: {i}') 

                    if eyetracker_type == 'gaze all':
                        list_df_left = []
                        list_df_right = []
                        list_df_ver = []
                        joined_patterns_set_left = set()
                        joined_patterns_set_right = set()
                        joined_patterns_set_ver = set()
                        
                        for i in os.listdir(base_path):
                            if i == '.DS_Store':
                                continue
                            
                            participant_dir = i  # Capture the participant directory
                            
                            # Define the patterns to search for
                            patterns = [
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/*/* Gaze-Left.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/* Gaze-Left.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/Outputs/* Gaze-Left.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/* Gaze-Left.csv'
                                ]

                            for pattern in patterns:
                                joined_patterns_set_left.add(pattern)

                        # Convert the set back to a list if needed
                        joined_patterns_left = list(joined_patterns_set_left)

                        # Use a dictionary to collect unique matches along with their participant directories
                        unique_matches = {}

                        for pattern in joined_patterns_left:
                            matches = glob.glob(pattern)
                            for match in matches:
                                # Extract the participant directory from the match
                                match_parts = match.split('/')
                                for part in match_parts:
                                    if part.startswith('participante_'):
                                        participant_dir = part
                                        break
                                unique_matches[match] = participant_dir

                        # Print the unique matches and create dataframes
                        for match, participant_dir in tqdm( unique_matches.items()):
                            df = pd.read_csv(match, delimiter=';')
                            df['Participant'] = participant_dir.split('_')[1]
                            df = impute_gaze_mv(df)
                            df = outlier_perclos(df,'replace')
                            df = create_open_eye(df)
                            df = real_perclos(df)
                            list_df_left.append(df)
      
                        
                        for i in os.listdir(base_path):
                            if i == '.DS_Store':
                                continue
                            
                            participant_dir = i  # Capture the participant directory
                            
                            # Define the patterns to search for
                            patterns = [
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/*/* Gaze-Right.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/* Gaze-Right.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/Outputs/* Gaze-Right.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/* Gaze-Right.csv'
                            ]

                            for pattern in patterns:
                                joined_patterns_set_right.add(pattern)

                        # Convert the set back to a list if needed
                        joined_patterns_right = list(joined_patterns_set_right)

                        # Use a dictionary to collect unique matches along with their participant directories
                        unique_matches = {}

                        for pattern in joined_patterns_right:
                            matches = glob.glob(pattern)
                            for match in matches:
                                # Extract the participant directory from the match
                                match_parts = match.split('/')
                                for part in match_parts:
                                    if part.startswith('participante_'):
                                        participant_dir = part
                                        break
                                unique_matches[match] = participant_dir

                        # Print the unique matches and create dataframes
                        for match, participant_dir in tqdm(unique_matches.items()):
                            df = pd.read_csv(match, delimiter=';')
                            df['Participant'] = participant_dir.split('_')[1]
                            df = impute_gaze_mv(df)
                            df = outlier_perclos(df,'replace')
                            df = create_open_eye(df)
                            df = real_perclos(df)
                            list_df_right.append(df)
                    
                    # if not found_match:
                    #     print(f'This one is not being counted: {i}')             
                    
                        for i in os.listdir(base_path):
                            if i == '.DS_Store':
                                continue
                            
                            participant_dir = i  # Capture the participant directory
                            
                            # Define the patterns to search for
                            patterns = [
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/*/* Gaze-Vergence.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/* Gaze-Vergence.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/*/Outputs/* Gaze-Vergence.csv',
                                f'{base_path}/{participant_dir}/dist_{participant_dir}_*_*_*/dist_{participant_dir}_eyetracker_*_*_*/* Gaze-Vergence.csv'
                                ]

                            for pattern in patterns:
                                joined_patterns_set_ver.add(pattern)

                        # Convert the set back to a list if needed
                        joined_patterns_ver = list(joined_patterns_set_ver)

                        # Use a dictionary to collect unique matches along with their participant directories
                        unique_matches = {}

                        for pattern in joined_patterns_ver:
                            matches = glob.glob(pattern)
                            for match in matches:
                                # Extract the participant directory from the match
                                match_parts = match.split('/')
                                for part in match_parts:
                                    if part.startswith('participante_'):
                                        participant_dir = part
                                        break
                                unique_matches[match] = participant_dir

                        # Print the unique matches and create dataframes
                        for match, participant_dir in tqdm(unique_matches.items()):
                            df = pd.read_csv(match, delimiter=';')
                            df['Participant'] = participant_dir.split('_')[1]
                            df = impute_gaze_mv(df)
                            df = outlier_perclos(df,'replace')
                            df = create_open_eye(df)
                            df = real_perclos(df)
                            list_df_ver.append(df)
                    # Check if any of the lists are empty and raise an error
                    if not list_df_left or not list_df_right or not list_df_ver:
                        raise ValueError("One or more of the dataframes lists are empty. Please check your input files and try again.")

                    return list_df_left, list_df_right, list_df_ver
                    
                    
                    #fixation
                    if eyetracker_type == 'fixation left':
                        for i in os.listdir(base_path):   
                            folder = f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*_Default_60sec Fixation-Left.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                        print('Dataframe created!')              
                    if eyetracker_type == 'fixation right':
                        for i in os.listdir(base_path):   
                            folder = f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*_Default_60sec Fixation-Right.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                        print('Dataframe created!')              
                    if eyetracker_type == 'fixation vergence':
                        for i in os.listdir(base_path):   
                            folder = f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_eyetracker_*_*_*/*_Default_60sec Fixation-Vergence.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                                print('Dataframe created!')              
            ##############################################################################################
            if experiment == 'distraction' and file_type == 'txt':
                for i in os.listdir(base_path): 
                    patterns = [
                    f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_simulador/dist_{i}_simulador_txt.txt',
                    f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_simulador/dist_{i}_simulador.txt'
                ]  
                    for pattern in patterns: 
                        folder = pattern.format(i=i)
                        matches = glob.glob(folder)
                        # Check if there is at least one match
                        if matches:
                            df, data = create_simulator_df(matches[0])
                            df['Participant'] = i.split('_')[1]
                            #append the dataframe to the empty dataframe
                            list_df.append(df)
                            print('Dataframe created!')  
            if experiment == 'distraction' and file_type == 'log':
                for i in os.listdir(base_path):   
                    patterns = [
                    f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_simulador/dist_{i}_simulador_telnav.log',
                    f'{base_path}/{i}/dist_{i}_*_*_*/dist_{i}_simulador/dist_{i}_simulador.log']
                    for pattern in patterns: 
                        folder = pattern.format(i=i)
                        matches = glob.glob(folder)
                        # Check if there is at least one match
                        if matches:
                            df = distraction_log(matches[0])
                            df['ID'] = i.split('_')[1]
                            #append the dataframe to the empty dataframe
                            list_df.append(df)
                            print('Dataframe created!')       

            ############################## DROWSINESS CSV ########################################
            if experiment == 'drowsiness' and file_type == 'csv':
                #prompt the user for additional information
                # print('Please provide the data source (e.g. "smartwatch"):')
                # data_source = input()
                #print('The data source chosen was: ' + data_source + '\n')
                if data_source == 'smartwatch': 
                    for participant_dir in os.listdir(base_path):   
                        folder = f'{base_path}/{participant_dir}/sono_{participant_dir}_*_*_*/sono_{participant_dir}_smartwatch_*_*_*/*.csv'
                        matches = glob.glob(folder)
                    if matches: 
                        for i in tqdm(os.listdir(base_path)):   
                            if i == '.DS_Store':
                                continue
                            # folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_smartwatch_*_*_*/*.csv'
                            # matches = glob.glob(folder)
                            # # Check if there is at least one match
                            # if matches:
                            
                            df_watch = pd.read_csv(matches[0])
                            #add a column to the df with the participant id
                            df_watch['Participant'] = i.split('_')[1]
                            #append the dataframe to the empty dataframe
                        
                            list_df.append(df_watch)
                                
                if data_source == 'eyetracker':
                    # print('Blink, Gaze or Fixation? Left, Right or Vergence? (e.g. "blink left"):')
                    # eyetracker_type = input()
                    #blink
                    if eyetracker_type == 'blink left':
                    
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Blink-Left.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                    if eyetracker_type == 'blink right':
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Blink-Right.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                        print('Dataframe created!')        
                    if eyetracker_type == 'blink vergence':
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Blink-Vergence.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                        print('Dataframe created!')          
                    #gaze
                    if eyetracker_type == 'gaze left':
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Left.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df = impute_gaze_mv(df)
                                df = outlier_perclos(df,'replace')
                                df = real_perclos(df)
                                df['Participant'] = i.split('_')[1]               
                                list_df.append(df)
                        print('Dataframe created!')         
                    if eyetracker_type == 'gaze right':
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Right.csv'
                            matches = glob.glob(folder)
                        
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                df = impute_gaze_mv(df)
                                df = outlier_perclos(df,'replace')
                                df = real_perclos(df)
                                list_df.append(df)
                                #list_df = sinc_dataframes(list_df, 'eyetracker')
                        print('Dataframe created!')           
                    if eyetracker_type == 'gaze vergence':
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Vergence.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                df = impute_gaze_mv(df)
                                #df = outlier_perclos(df,'replace')
                                list_df.append(df)
                                #list_df = sinc_dataframes(list_df, 'eyetracker')
                                #list_df = [df.drop('perclos', axis=1) for df in list_df]
                        print('Dataframe created!')   
                    if eyetracker_type == 'gaze all':
                        list_df_left = []
                        list_df_right = []
                        list_df_ver = []
                        print('Creating dataframes for gaze left...')
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Left.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df = impute_gaze_mv(df)
                                df = outlier_perclos(df,'replace')
                                df = create_open_eye(df)
                                df = real_perclos(df)
                                df['Participant'] = i.split('_')[1]               
                                list_df_left.append(df)
                                #list_df_left = sinc_dataframes(list_df_left, 'eyetracker')
                                #list_df_left = [df.drop('perclos', axis=1) for df in list_df_left]
                        print('Creating dataframes for gaze right...')
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Right.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                df = impute_gaze_mv(df)
                                df = outlier_perclos(df,'replace')
                                df = create_open_eye(df)
                                df = real_perclos(df)
                                list_df_right.append(df)
                                #list_df_right = sinc_dataframes(list_df_right, 'eyetracker')
                                #list_df_right = [df.drop('perclos', axis=1) for df in list_df_right]
                        print('Creating dataframes for gaze vergence...')
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Gaze-Vergence.csv'
                            matches = glob.glob(folder)
                        
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                df = impute_gaze_mv(df)
                                df = outlier_perclos(df,'replace')
                                list_df_ver.append(df)
                                #list_df_ver = sinc_dataframes(list_df_ver, 'eyetracker')
                                #list_df_ver = [df.drop('perclos', axis=1) for df in list_df_ver]
                        return list_df_left, list_df_right, list_df_ver

                    #fixation
                    if eyetracker_type == 'fixation left':
                        
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Fixation-Left.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                                 
                    if eyetracker_type == 'fixation right':
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Fixation-Right.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                                  
                    if eyetracker_type == 'fixation vergence':
                        for i in tqdm(os.listdir(base_path)):   
                            folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_eyetracker_*_*_*/*/Outputs/* Fixation-Vergence.csv'
                            matches = glob.glob(folder)
                            if matches: 
                                df = pd.read_csv(matches[0], delimiter = ';')
                                df['Participant'] = i.split('_')[1]
                                list_df.append(df)
                                 
        ##########################################################################################
            if experiment == 'drowsiness' and file_type == 'txt':
                for participant_dir in os.listdir(base_path):
                    if participant_dir == '.DS_Store':
                        continue
                    glob_pattern = os.path.join(base_path, participant_dir, 'sono_{}_{}_{}_*', 'sono_{}_simulador', '*.txt')
                    glob_path = glob_pattern.format(participant_dir, '*', '*', participant_dir, participant_dir)
                    matches = glob.glob(glob_path)
                if matches: 
                    print('Extracting simulator files...')
                    for participant_dir in tqdm(os.listdir(base_path)):
                        if participant_dir == '.DS_Store':
                            continue
                        # Assuming the structure is consistent and there's only one "sono" directory per participant
                        # glob_pattern = os.path.join(base_path, participant_dir, 'sono_{}_{}_{}_*', 'sono_{}_simulador', '*.txt')
                        # glob_path = glob_pattern.format(participant_dir, '*', '*', participant_dir, participant_dir)
                        # matches = glob.glob(glob_path)
                        
                        # if matches:
                        df, df_dict = create_simulator_df(matches[0])
                        df['Participant'] = participant_dir.split('_')[1]
                        df = create_angv(df)
                        df = calculate_interactions(df)
                        df = std_dev(df, 'vel(km/h)')
                        df = std_dev(df, 'eixo_offset(m)')
                        df = std_dev(df, 'acelerador(per)')
                        df = std_dev(df, 'vel_ang(rad/s)')
                        df = pisa_linha_reset(df)
                        df = muda_faixa(df)
                        list_df.append(df)
                                  
            if experiment == 'drowsiness' and file_type == 'log':
                for participant_dir in os.listdir(base_path):
                    if participant_dir == '.DS_Store':
                        continue
                    folder = f'{base_path}/{participant_dir}/sono_{participant_dir}_*_*_*/sono_{participant_dir}_simulador/*.log'
                    matches = glob.glob(folder)
                if matches:
                    print('Extracting telnav files...')
                    for i in tqdm(os.listdir(base_path)):  
                        if participant_dir == '.DS_Store':
                            continue 
                        folder = f'{base_path}/{i}/sono_{i}_*_*_*/sono_{i}_simulador/*.log'
                        matches = glob.glob(folder)
                        # Check if there is at least one match
                        try:
                            if matches:
                                df = create_df(matches[0], experiment, i)
                                df = fix_timestamps(df)
                                list_df.append(df)
                                        
                        #if a key error occurs, print the index of the participant
                        except KeyError:
                            print(f'Key error occurred on{i}')
                            
                       

        #if df_dict is defined, return it as well
            if 'df_dict' in locals():
                return list_df, df_dict
            else:
                return list_df
        except ValueError:
            print('No files found in the selected directory. Please try again.')
            base_path = select_base_path()
            continue            

def full_extraction():
    base_path = select_base_path()
    print('Extracting simulator files...')
    sim_log = extract_all('drowsiness', 'simulator', None, 'log', base_path)
    sim_txt, sim_dict = extract_all('drowsiness','simulator', None,  'txt',base_path)
    print('Extracting eyetracker files...')
    list_df_ver_left, list_df_ver_right, list_df_ver_ver = extract_all('drowsiness', 'eyetracker', 'gaze all', 'csv', base_path)
    print('Extracting smartwatch files...')
    smartwatch = extract_all('drowsiness', 'smartwatch', None, 'csv',base_path)
    smartwatch = sinc_dataframes(smartwatch, 'smartwatch')
    smartwatch = prepare_smartwatch(smartwatch)

    return sim_log, sim_txt, sim_dict, list_df_ver_left, list_df_ver_right, list_df_ver_ver, smartwatch





def join_vergences(left, right, ver):  #try this 
    #if vergence[0] has column 'perclos'
    try:
        if 'pupil_percentage' in ver[0].columns:
            ver = [df.drop('pupil_percentage', axis=1) for df in ver]
    except IndexError:
        print('No perclos column found in the vergence dataframes. Try again')

    list_of_dfs = []
    for df1 in left:
        for df2 in right:
            for df3 in ver:
            # Check if the DataFrames are not empty and the required columns exist
                if not df1.empty and not df2.empty and not df3.empty and \
                    'Participant' in df1.columns and 'Participant' in df2.columns and 'Participant' in df3.columns and \
                    df1['Participant'].iloc[0] == df2['Participant'].iloc[0] == df3['Participant'].iloc[0]:
                    # Assuming 'participant' and 'id' are unique within each DataFrame,
                    # and there's only one row per participant/id in each DataFrame.
                    participant_id1 = df1['Participant'].iloc[0]
                    participant_id2 = df2['Participant'].iloc[0]
                    if participant_id1 == participant_id2:
                        df1.sort_values(by='Start Time (secs)', inplace=True)
                        df2.sort_values(by='Start Time (secs)', inplace=True)
                        df3['perclos left'] = df1['perclos']
                        df3['perclos right'] = df2['perclos']
                        df3['perclos mean'] = df3[['perclos left', 'perclos right']].mean(axis=1)
                        df3['pupil percentage left'] = df1['pupil_percentage']
                        df3['pupil percentage right'] = df2['pupil_percentage']
                        df3['pupil percentage mean'] = df3[['pupil percentage left', 'pupil percentage right']].mean(axis=1)
                        df3['left eye open'] = df1['eye open']
                        df3['right eye open'] = df2['eye open']
                        df3['both eyes open'] = np.where((df3['left eye open'] == 1) & (df3['right eye open'] == 1), 1, 0)
                        df3['left pupil diameter'] = df1['Pupil Diameter']
                        df3['right pupil diameter'] = df2['Pupil Diameter']
                        df3['mean pupil diameter'] = df3[['left pupil diameter', 'right pupil diameter']].mean(axis=1)
                      #delete rows with null values from ver
                    for i in range(len(ver)):
                        df3.dropna(inplace=True)
                    list_of_dfs.append(df3)
           
    return list_of_dfs


def join_log(list1, list2): #merges vergence with log simulator data
    #list1 - vergence
    #list2 - log
    matching_pairs = []  # Todf = pd.DataFrame()  # To store the merged DataFrames
    merged = []
    for df1 in list1:
        for df2 in list2:
            # Check if the DataFrames are not empty and the required columns exist
            if not df1.empty and not df2.empty and \
               'Participant' in df1.columns and 'ID' in df2.columns:
                # Assuming 'participant' and 'id' are unique within each DataFrame,
                # and there's only one row per participant/id in each DataFrame.
                if df1['Start Time (secs)'].dtype == 'object':
                    df1['Start Time (secs)'] = df1['Start Time (secs)'].str.replace(',', '.').astype(float)

                df1['Start Time (secs)'] = pd.to_numeric(df1['Start Time (secs)'], errors='coerce').astype('float64')
                if df2['Timestamp'].dtype == 'object':
                    df2['Timestamp'] = df2['Timestamp'].str.replace(',', '.').astype(float)
                
                df2['Timestamp'] = pd.to_numeric(df2['Timestamp'], errors='coerce').astype('float64')
                participant_id1 = df1['Participant'].iloc[0]
                participant_id2 = df2['ID'].iloc[0]
                if participant_id1 == participant_id2:
                    df1.sort_values(by='Start Time (secs)', inplace=True)
                    df2.sort_values(by='Timestamp', inplace=True)
                    matching_pairs.append((df1, df2))
                    merged_df = pd.merge_asof(df1, df2, left_on='Start Time (secs)', right_on='Timestamp', direction='forward')
                    #delete collumns that are not needed
                    merged_df = merged_df.drop(['Timestamp', 'ID'], axis=1)
                    merged.append(merged_df)
                    
    
    return merged


##############################################################################################################



def prepare_smartwatch(list_df):
    for df in list_df:
        if 'secs' not in df.columns or 'hr' not in df.columns:
            # If any DataFrame lacks 'secs' or 'hr' columns, return the original list
            return list_df

    # Perform the transformations if all DataFrames have 'secs' and 'hr' columns
    list_df = [df[['Participant', 'secs', 'hr']] for df in list_df]
    list_df = [df.rename(columns={'secs': 'tempo(s)', 'hr': 'Heart Rate'}) for df in list_df]
    list_df = [df.astype({'tempo(s)': 'float64'}) for df in list_df]

    return list_df


def merge_sims(list1, list2):

    matching_pairs = []
    merged = []

    for df1 in list1:
        for df2 in list2:
            if not df1.empty and not df2.empty and \
                'Participant' in df1.columns and 'ID' in df2.columns:
                
                participant_id1 = df1['Participant'].iloc[0]
                participant_id2 = df2['ID'].iloc[0]
                
                if participant_id1 == participant_id2:
                    if df2['Timestamp'].dtype == 'object':
                        df2['Timestamp'] = df2['Timestamp'].str.replace(',', '.').astype(float)
                    if df1['tempo(s)'].dtype == 'object':
                        df1['tempo(s)'] = df1['tempo(s)'].str.replace(',', '.').astype(float)
                    # Convert to numeric, coercing errors to NaN
                    #df1['tempo(s)'] = pd.to_numeric(df1['tempo(s)'], errors='coerce')
                    df2['Timestamp'] = pd.to_numeric(df2['Timestamp'], errors='coerce')
                    
                    # Convert to int64 (if appropriate, use floor or ceil if fractional parts are not desired)
                    #df1['tempo(s)'] = df1['tempo(s)'].astype('int64')
                    #df2['Timestamp'] = df2['Timestamp'].astype('int64')

                

                    # df1['tempo(s)'] = pd.to_numeric(df1['tempo(s)'], errors='coerce')
                    df2['Timestamp'] = pd.to_numeric(df2['Timestamp'], errors='coerce')
                    
                    df1.sort_values(by='tempo(s)', inplace=True)
                    df2.sort_values(by='Timestamp', inplace=True)
                    
                    # Check for NaN values after conversion and sorting
                    if df1['tempo(s)'].isnull().any() or df2['Timestamp'].isnull().any():
                        print('NaN values detected in the timestamp columns after conversion')
                    
                    matching_pairs.append((df1, df2))
                    merged_df = pd.merge_asof(df1, df2, left_on='tempo(s)', right_on='Timestamp', direction='forward')
                    
                    # Delete columns that are not needed
                    merged_df = merged_df.drop(['Timestamp', 'ID'], axis=1)
                    merged.append(merged_df)
        
    return merged


def join_sim_w_smartwatch(sim, smartwatch): #ver = sim; 
    merged = []
    for df1 in sim:
        for df2 in smartwatch:
            # Ensure the 'Participant' column exists in both DataFrames and has matching first entries
            if not df1.empty and not df2.empty and \
            'Participant' in df1.columns and 'Participant' in df2.columns and \
            df1['Participant'].iloc[0] == df2['Participant'].iloc[0]:
                participant_id1 = df1['Participant'].iloc[0]
                participant_id2 = df2['Participant'].iloc[0]
                if participant_id1 == participant_id2:
                    # Ensure sorting is done on 'Start Time (secs)'
                    
                    df1.sort_values(by='tempo(s)', inplace=True)

                    # Convert 'tempo(s)' in df2 to numeric values, handling non-numeric values as NaN
                    df2['tempo(s)'] = pd.to_numeric(df2['tempo(s)'], errors='coerce')
                    df2.dropna(subset=['tempo(s)'], inplace=True)  # Optional: drop rows where 'tempo(s)' is NaN
                    df2.sort_values(by='tempo(s)', inplace=True)
                    
                    # Now perform the merge_asof
                    merged_df = pd.merge_asof(df1, df2, left_on='tempo(s)', right_on='tempo(s)', direction='forward')
                    

                    
                    merged_df['Heart Rate'] = merged_df['Heart Rate'].interpolate(method='linear')

                    merged.append(merged_df)
    merged = [df_merged.drop('Participant_y', axis=1) for df_merged in merged]
    merged = [df_merged.rename(columns={'Participant_x': 'Participant'}) for df_merged in merged]
            
    return merged

def rename_columns_in_list(df_list, rename_dict):
    for i in range(len(df_list)):
        df_list[i] = df_list[i].rename(columns=rename_dict)
    return df_list

def join_other(list1, list2): #ver = sim; 
    merged = []

    # Define a dictionary with possible column names as keys and the desired column name as the value
    rename_dict = {
        'id': 'Participant',
        'ID': 'Participant',
        'participant': 'Participant',
        'participant_x': 'Participant'
    }

    list2 = rename_columns_in_list(list2, rename_dict)
    for df1 in list1:
        for df2 in list2:
            #convert df1['Participant'] to int64
            df1['Participant'] = pd.to_numeric(df1['Participant'], errors='coerce')


            # Ensure the 'Participant' column exists in both DataFrames and has matching first entries
            if not df1.empty and not df2.empty and \
            'Participant' in df1.columns and 'Participant' in df2.columns and \
            df1['Participant'].iloc[0] == df2['Participant'].iloc[0]:
                participant_id1 = df1['Participant'].iloc[0]
                participant_id2 = df2['Participant'].iloc[0]
                if participant_id1 == participant_id2:

                    #if df1 has column tempo(s)_x, rename it to tempo(s)
                    if 'tempo(s)_x' in df1.columns:
                        df1 = df1.rename(columns={'tempo(s)_x': 'tempo(s)'})
                    

                    # Rename the columns in df2 based on the rename_dict
                    df2 = df2.rename(columns=rename_dict)

                    #if df2['tempo(s)] is type object or string
                    if df2['tempo(s)'].dtype == 'object' or df2['tempo(s)'].dtype == 'string':
                        df2['tempo(s)'] = df2['tempo(s)'].str.replace(',', '.').astype(float)
                    #convert df2[tempo(s)] to float
                    df2 = df2.astype({'tempo(s)': 'float64'})
                    # Ensure sorting is done on 'Start Time (secs)'
                    
                    df1.sort_values(by='tempo(s)', inplace=True)

                    # Convert 'tempo(s)' in df2 to numeric values, handling non-numeric values as NaN
                    df2['tempo(s)'] = pd.to_numeric(df2['tempo(s)'], errors='coerce')
                    df2.dropna(subset=['tempo(s)'], inplace=True)  # Optional: drop rows where 'tempo(s)' is NaN
                    df2.sort_values(by='tempo(s)', inplace=True)
                    
                    # Now perform the merge_asof
                    merged_df = pd.merge_asof(df1, df2, left_on='tempo(s)', right_on='tempo(s)', direction='forward')

                    merged.append(merged_df)
                else:
                    print('Participant IDs do not match - this might lead to problems in the merging of datasets')
            else: 
                print('There is not a column called ID/Participant on the additional dataframe! Please correct this.')
    merged = [df_merged.drop('Participant_y', axis=1) for df_merged in merged]
    merged = [df_merged.rename(columns={'Participant_x': 'Participant'}) for df_merged in merged]
            

    return merged




def join_smartwatch_txt(ver, other_df, sampling='downsample'): #other_df can either be smartwatch data or txt sim data
    merged = []
    #other_df = [df['tempo(s)'].str.replace(',', '.').astype(float) for df in other_df]
    #other_df = [df.astype({'tempo(s)': 'float64'}) for df in other_df]
    ver = [df.astype({'Start Time (secs)': 'float64'}) for df in ver]
    if sampling == 'downsample':
        merged_list = []
        for df1 in ver:
            for df2 in other_df:
                # Ensure the 'Participant' column exists in both DataFrames and has matching first entries
                if not df1.empty and not df2.empty and \
                'Participant' in df1.columns and 'Participant' in df2.columns and \
                df1['Participant'].iloc[0] == df2['Participant'].iloc[0]:
                    
                    participant_id1 = df1['Participant'].iloc[0]
                    participant_id2 = df2['Participant'].iloc[0]
                    if participant_id1 == participant_id2:
                        if df2['tempo(s)'].dtype == 'object':
                            df2['tempo(s)'] = df2['tempo(s)'].str.replace(',', '.').astype(float)
                        df2['tempo(s)'] = pd.to_numeric(df2['tempo(s)'], errors='coerce')
                        
                        merged_df = merge_on_closest_timestamp_downsample(df1, df2)
                        merged_list.append(merged_df)

        return merged_list
        
    else:
        
        for df1 in ver:
            for df2 in other_df:
                # Ensure the 'Participant' column exists in both DataFrames and has matching first entries
                if (
                    not df1.empty and not df2.empty and 
                    'Participant' in df2.columns and 
                    ('Participant' in df1.columns or 'Participant_x' in df1.columns) and 
                    df2['Participant'].iloc[0] == (df1['Participant'].iloc[0] if 'Participant' in df1.columns else df1['Participant_x'].iloc[0])
                ):

                    
                    if 'Participant_x' in df1.columns:
                        #rename column to 'Participant'
                        df1 = df1.rename(columns={'Participant_x': 'Participant'})
                    participant_id1 = df1['Participant'].iloc[0]
                    participant_id2 = df2['Participant'].iloc[0]
                    if participant_id1 == participant_id2:
                        if df2['tempo(s)'].dtype == 'object':
                            df2['tempo(s)'] = df2['tempo(s)'].str.replace(',', '.').astype(float)
                        df2['tempo(s)'] = pd.to_numeric(df2['tempo(s)'], errors='coerce')
                        
                        # Ensure sorting is done on 'Start Time (secs)'
                        df1.sort_values(by='Start Time (secs)', inplace=True)
                        df2.dropna(subset=['tempo(s)'], inplace=True)  # Optional: drop rows where 'tempo(s)' is NaN
                        df2.sort_values(by='tempo(s)', inplace=True)
                        
                        # Now perform the merge_asof
                        merged_df = pd.merge_asof(df1, df2, left_on='Start Time (secs)', right_on='tempo(s)', direction='forward')
                        

                        # for column in columns_to_interpolate:
                        #     if column in merged_df.columns:
                        #         merged_df[column] = merged_df[column].interpolate(method='linear')

                        merged.append(merged_df)
        merged = [df_merged.drop('Participant_y', axis=1) for df_merged in merged]
        merged = [df_merged.rename(columns={'Participant_x': 'Participant'}) for df_merged in merged]
                
        return merged

def merge_on_closest_timestamp_downsample(df1, df2):
    """
    Efficiently merges two dataframes by finding the closest timestamp in df1 for each entry in df2.

    Parameters:
    - df1: DataFrame with a 'Start Time (secs)' column.
    - df2: DataFrame with a 'tempo(s)' column.

    Returns:
    - Merged DataFrame with closest timestamps.
    """

    # Convert columns to numeric, handling non-numeric gracefully
    df1['Start Time (secs)'] = pd.to_numeric(df1['Start Time (secs)'], errors='coerce')
    df2['tempo(s)'] = pd.to_numeric(df2['tempo(s)'], errors='coerce')

    # Sort dataframes by timestamp
    df1 = df1.sort_values(by='Start Time (secs)')
    df2 = df2.sort_values(by='tempo(s)')

    # Use merge_asof to merge on the closest timestamp
    merged_df = pd.merge_asof(df2, df1, left_on='tempo(s)', right_on='Start Time (secs)', direction='nearest')
    
    return merged_df



    ##############################################################################################################



def find_equal_sequences(df, column_name):
    """
    This function finds sequences of equal values in a specified column of a DataFrame
    and returns a list of tuples with the indexes of the first and last element of each sequence.

    Parameters:
    - df: Pandas DataFrame
    - column_name: String, the name of the column to search for sequences

    Returns:
    - List of tuples, each tuple containing the indexes (start, end) of each sequence
    """
    # Initialize an empty list to store the sequences
    sequences = []
    
    # Initialize the start index of the current sequence to None
    start_index = None
    
    # Iterate over the DataFrame using itertuples() for efficiency
    for i, value in enumerate(df[column_name]):
        # Check if we are at the start of a sequence
        if start_index is None:
            start_index = i
        else:
            # If the current value does not match the previous one, or it's the last element,
            # we have reached the end of a sequence
            if value != df[column_name].iloc[i - 1] or i == len(df[column_name]) - 1:
                # If it's the last element and still part of the current sequence, adjust the end index
                end_index = i if value == df[column_name].iloc[i - 1] else i - 1
                
                # Add the sequence to the list if it's length is more than 1
                if end_index > start_index:
                    sequences.append((start_index, end_index))
                
                # Reset the start index for the next sequence
                start_index = i if value != df[column_name].iloc[i - 1] else None
    
    return sequences


def linear_interpolation(df, sequences):
    """
    Applies a specified function to each sequence in the DataFrame for multiple columns.

    Parameters:
    - df: Pandas DataFrame
    - sequences: List of tuples, each tuple containing the start and end index of a sequence
    - columns: List of strings, each representing a column name to apply the function
   
    Returns:
    - A DataFrame with the specified function applied to each sequence for the given columns.
    """
    columns = ['X',
 'Y',
 'roaddist(m)',
 'totaldistance(m)',
 'vel(km/h)',
 'eixo_offset(m)',
 'ang_rodas(?)',
 'ang_carro(rad)',
 'ang_carro_via(rad)',
 'acelerador(per)',
 'travao(per)',
 'volante(g)',
 'vol(filt)',
 'dist_frente(m)',
 'vel_frente(km/h)',
 'TTC(s)',
 'TH(s)',
 'DH(m) )']
    df_copy = df.copy()
    for column in columns:  # Loop through each specified column
        for start, end in sequences:
            # Ensure the sequence has more than one element to interpolate
            if start != 0:
                # Get the start and end values of the sequence for the current column
                start_value = df_copy.loc[start - 1, column]
                end_value = df_copy.loc[end, column]
                # Calculate the difference between the start and end values
                diff = float(end_value) - float(start_value)
                # Calculate the number of steps between the start and end values
                steps = float(end) - (float(start) - 1) 
                
                # Calculate the increment to apply to each step
                increment = diff / steps
                
                # Apply the increment to each step in the sequence
                for index in range(start, end):
                    df_copy.loc[index, column] = float(df_copy.loc[index - 1, column]) + increment
    
    return df_copy





def join_sono_auto(interpolation, number_dfs = None):
    final_df_list = []
    print('Eyetracker preparations...')
    list_df_ver_left, list_df_ver_right, list_df_ver_ver = extract_all('drowsiness', 'eyetracker', 'gaze all', 'csv')



    print('Smartwatch preparations...')
    smartwatch = extract_all('drowsiness', 'smartwatch', None, 'csv')
    smartwatch = sinc_dataframes(smartwatch, 'smartwatch')
    smartwatch = prepare_smartwatch(smartwatch)

    print('Simulator preparations...')
    sim_log = extract_all('drowsiness', 'simulator', None, 'log')
    sim_txt, sim_dict = extract_all('drowsiness','simulator', None,  'txt')

    print('Joining dataframes...')
    ver = join_vergences(list_df_ver_left, list_df_ver_right, list_df_ver_ver)
    sinc_ver = sinc_dataframes(ver, 'eyetracker')
    merged_df = join_log(sinc_ver, sim_log)

    if number_dfs is not None: 
        merged_df = merged_df[0:number_dfs]
        sim_txt = sim_txt[0:number_dfs]

    merged_df_txt = join_smartwatch_txt(merged_df, sim_txt, interpolation)
    merged_full = join_smartwatch_txt(merged_df_txt, smartwatch, 'other')

    for df in merged_full: 
        merged_dataframes = cut_dataframe(df)
        final_df_list.append(merged_dataframes)

    return final_df_list, sim_dict 


#manual timestamp sinc (for our data we have a csv with the data for the sinc - as seen in function sinc_dataframes)
def seconds_since_midnight(dt):
    midnight = datetime(year=dt.year, month=dt.month, day=dt.day)
    return (dt - midnight).total_seconds()

def manual_sinc(df, device):
    eyetracker_start = input('Input the eyetracker start time (dd/mm/yyyy hh:mm:ss): ')
    eyetracker_start = datetime.strptime(eyetracker_start, '%d/%m/%Y %H:%M:%S')
    smartwatch_start = input('Input the smartwatch start time (dd/mm/yyyy hh:mm:ss): ')
    smartwatch_start = datetime.strptime(smartwatch_start, '%d/%m/%Y %H:%M:%S')
    sim_start = input('Input the simulation start time (%d-mm-yyyy hh:mm:ss): ')
    sim_start = datetime.strptime(sim_start, '%d/%m/%Y %H:%M:%S')
    et_seconds = seconds_since_midnight(eyetracker_start)
    sw_seconds = seconds_since_midnight(smartwatch_start)
    sim_seconds = seconds_since_midnight(sim_start)

    difference_et = et_seconds - sim_seconds
    difference_sw = sw_seconds - sim_seconds
    df = df.copy()
    if re.match('^eyetracker$', device, re.IGNORECASE):
        df['Start Time (secs)'] = df['Start Time (secs)'] - difference_et
    elif re.match('^smartwatch$', device, re.IGNORECASE):
        df['secs'] = df['secs'] - difference_sw
    else:
        print('Invalid device - must be either eyetracker or smartwatch')
    return difference_et, difference_sw


   
# def join_sono_manual():
#     columns = ['roaddist(m)','totaldistance(m)', 'vel(km/h)']
#     final_df_list = []
#     list_df_sw = extract_all('drowsiness', 'csv')
#     list_df_sw = sinc_dataframes(list_df_sw, 'smartwatch')
#     list_df_txt, txt_dict = extract_all('drowsiness', 'txt')
#     list_df_log = extract_all('drowsiness', 'log')
#     list_df_ver_left, list_df_ver_right, list_df_ver_ver = extract_all('drowsiness', 'csv')
#     list_df_sw = prepare_smartwatch(list_df_sw)
#     ver = join_vergences(list_df_ver_left, list_df_ver_right, list_df_ver_ver)
#     ver = sinc_dataframes(ver, 'eyetracker') 
#     merged_df = join_log(ver, list_df_log)
#     merged_df_txt = join_smartwatch_txt(merged_df, list_df_txt)
#     merged_full = join_smartwatch_txt(merged_df_txt, list_df_sw)
#     for df in merged_full:
#         sequences = find_equal_sequences(df, 'tempo(s)_x')
#         df_interpolated = linear_interpolation(df, sequences, columns)
#         final_df_list.append(df_interpolated)

#     return final_df_list, txt_dict 


def cut_dataframe(df):
    """
    Find the row index of the last non-0 and non-null value for the given features in a DataFrame.

    Parameters:
    - df: pandas DataFrame.
    - features: List of column names to check.

    Returns:
    - int: The index of the row containing the last non-0 and non-null value among the given features.
           Returns -1 if no such value is found.

    """
    #delete lines where tempo(s) is < 0
    df = df[df['tempo(s)_x'] >= 0]

    #reset index 
    df.reset_index(drop = True, inplace = True)


    #interações: gear, piscas, vel, acelerador, travao, volante
    features = ['gear', 'piscas(2=Dtr,4=Esq,5=QtPisc)', 'acelerador(per)', 'travao(per)']

    # Assuming sim_txt[5] is your DataFrame
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    #features = ['Feature_2','Feature_3']
    last_non_zero_non_null_idx = -1
    for feature in features:
        # Drop rows where the feature is NaN or 0, then get the last index
        valid_idx = df[df[feature].notna() & (df[feature] != 0)].index
        if not valid_idx.empty:
            max_idx = valid_idx.max()
            last_non_zero_non_null_idx = max(last_non_zero_non_null_idx, max_idx)

    #delete rows after the last non-zero and non-null value
    df = df.drop(df.index[last_non_zero_non_null_idx+1:])
    # List of columns to drop
    columns_to_drop = ['Start Time (secs)', 'tempo(s)_y', 'Participant_df2', 'Z', 'ACC_sate', 'Accspeed(km/h)', 'ACC_dist(s)']

    # Check if 'Participant_df2' is in the DataFrame columns
    if 'Participant_df2' in df.columns:
        df = df.drop(columns=columns_to_drop, axis=1)
    else:
        # Drop the same columns, excluding 'Participant_df2'
        columns_to_drop.remove('Participant_df2')
        df = df.drop(columns=columns_to_drop, axis=1)

    #rename column tempo
    df = df.rename(columns = {'tempo(s)_x':'timestamp'})

    #delete rows with null values
    df = df.dropna() #devera ser alterado mais tarde
    
    
    return df


def one_hot(df):

    df['piscas(2=Dtr,4=Esq,5=QtPisc)'] = pd.to_numeric(df['piscas(2=Dtr,4=Esq,5=QtPisc)'], errors='coerce')

    # Create the encoder
    encoder = OneHotEncoder(sparse=False, dtype=np.uint8)
    
    # Fit the encoder so it learns about the unique values
    encoder.fit(df[['piscas(2=Dtr,4=Esq,5=QtPisc)']])
    
    # Transform the 'Feature' column
    one_hot_encoded_data = encoder.transform(df[['piscas(2=Dtr,4=Esq,5=QtPisc)']])
    
    # Get the unique values in the order the encoder is using them
    # and create column names based on these unique values
    unique_values = encoder.categories_[0]
    column_names = [f'pisca_{int(value)}' for value in unique_values]
    
    # Create a DataFrame with the custom column names
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=column_names)
    
    # Concatenate the new DataFrame with the original one, excluding the original 'Feature' column
    result_df = pd.concat([df.drop(columns=['piscas(2=Dtr,4=Esq,5=QtPisc)']), one_hot_encoded_df], axis=1)
    return result_df

def create_sleepy(df):
    #create column sleepy in final_0
    df['sleepy'] = 0
    #if kss > 6, then sleepy = 1
    df.loc[df['KSS'] >= 6, 'sleepy'] = 1
    return df

# def std_dev(df, feature):

    
#     # Assuming 'df' is your DataFrame
#     df['tempo(s)'] = df['tempo(s)'].astype(float)

#     # Create bins of 60 seconds each from the minimum to the maximum time in 'tempo(s)'
#     bins = range(int(df['tempo(s)'].min()), int(df['tempo(s)'].max()) + 60, 60)

#     # Use pd.cut to assign each time in 'tempo(s)' to a 60-second bin
#     df['time_bin'] = pd.cut(df['tempo(s)'], bins, right=False, labels=False)

#     # Group by the new time_bin column and calculate the standard deviation for 'vel(km/h)'
#     velocity_std_per_60s = df.groupby('time_bin')[feature].std()

#     # Optionally, create a mapping to the original DataFrame (if you need it in the original DataFrame)
#     df[f'{feature}_std_per_60s'] = df['time_bin'].map(velocity_std_per_60s)

#     return df


def std_dev(df, feature):
    
    window_size = 3
    
    # Convert 'tempo(s)' to float if not already
    df['tempo(s)'] = df['tempo(s)'].astype(float)
    
    # Sort the DataFrame by 'tempo(s)' to ensure the rolling window follows the time sequence
    df = df.sort_values('tempo(s)')
    
    # Calculate rolling standard deviation with the specified window size (in seconds)
    # Assuming 'tempo(s)' is in seconds and indexed appropriately
    df[f'{feature}_rolling_std'] = df[feature].rolling(window=window_size, min_periods=1).std()
    
    # Replace NaN values in the rolling std dev column with 0
    df[f'{feature}_rolling_std'] = df[f'{feature}_rolling_std'].fillna(0)
    
    return df


def degrees_to_radians(degrees):
    radians = degrees * (math.pi / 180)
    return radians

# def create_angv(df):
#     #create column volante(rad)
#     df['volante(rad)'] = 0.0
#     df['diff vol(rad)' ] = 0.0
#     df['diff tempo(s)'] = 0.0
#     df['vel_ang(rad/s)'] = 0.0

#     df['tempo(s)'] = pd.to_numeric(df['tempo(s)'], errors='coerce')


#     for i in range(len(df)):
#         # Convert and assign radians
#         df.loc[i, 'volante(rad)'] = degrees_to_radians(float(df.loc[i, 'volante(g)']))

#         # Calculate differences and handle the first row
#         if i > 0:
#             df.loc[i, 'diff vol(rad)'] = df.loc[i, 'volante(rad)'] - df.loc[i - 1, 'volante(rad)']
#             df.loc[i, 'diff tempo(s)'] = df.loc[i, 'tempo(s)'] - df.loc[i - 1, 'tempo(s)']
#             # Calculate angular velocity, handling division by zero
#             if df.loc[i, 'diff tempo(s)'] != 0:
#                 df.loc[i, 'vel_ang(rad/s)'] = df.loc[i, 'diff vol(rad)'] / df.loc[i, 'diff tempo(s)']
#         else:
#             df.loc[i, 'diff vol(rad)'] = 0
#             df.loc[i, 'diff tempo(s)'] = 0
#             df.loc[i, 'vel_ang(rad/s)'] = 0  # Set to 0 or NaN based on what makes sense in your context
#     return df

def create_angv(df):
    # Convert 'tempo(s)' column to numeric, coerce errors to NaN
    df['tempo(s)'] = pd.to_numeric(df['tempo(s)'], errors='coerce')

    # Convert 'volante(g)' to radians and create a new column 'volante(rad)'
    df['volante(rad)'] = degrees_to_radians(df['volante(g)'].astype(float))

    # Calculate the differences using vectorized operations
    df['diff vol(rad)'] = df['volante(rad)'].diff().fillna(0)
    df['diff tempo(s)'] = df['tempo(s)'].diff().fillna(0)

    # Calculate angular velocity, handling division by zero
    df['vel_ang(rad/s)'] = np.where(df['diff tempo(s)'] != 0, df['diff vol(rad)'] / df['diff tempo(s)'], 0)

    return df

# def calculate_interactions(df):
#     df['interval_cumulative_interactions'] = 0
#     interval_interactions = 0
#     start_interval_time = df['tempo(s)'].iloc[0]
#     df['volante(g)'] = pd.to_numeric(df['volante(g)'], errors='coerce')


#     for index, row in df.iterrows():
#         current_time = row['tempo(s)']

#         # Check if current time has exceeded the 60-second interval
#         if current_time >= start_interval_time + 60:
#             interval_interactions = 0  # Reset the counter
#             start_interval_time = current_time  # Update start time for the new interval

#         if index > 0:
#             changes_travao = ((row['travao(per)'] != df.at[index - 1, 'travao(per)']) and (row['travao(per)'] != 0))
#             changes_acelerador = ((row['acelerador(per)'] != df.at[index - 1, 'acelerador(per)']) and (row['acelerador(per)'] != 0))
#             gear_changes = (row['gear'] != df.at[index - 1, 'gear'])
#             volante_changes = abs(row['volante(g)'] - df.at[index - 1, 'volante(g)']) > 6  # Additional condition for ang_volante

#             interactions_count = int(changes_travao) + int(changes_acelerador) + int(gear_changes) + int(volante_changes)
#         else:
#             interactions_count = 0  # No interactions for the first row

#         interval_interactions += interactions_count
#         df.at[index, 'interval_cumulative_interactions'] = interval_interactions

#    return df


def calculate_interactions(df):
    df['volante(g)'] = pd.to_numeric(df['volante(g)'], errors='coerce')
    df['interval_cumulative_interactions'] = 0

    # Calculate changes for each condition
    df['changes_travao'] = (df['travao(per)'] != df['travao(per)'].shift()) & (df['travao(per)'] != 0)
    df['changes_acelerador'] = (df['acelerador(per)'] != df['acelerador(per)'].shift()) & (df['acelerador(per)'] != 0)
    df['gear_changes'] = df['gear'] != df['gear'].shift()
    df['volante_changes'] = abs(df['volante(g)'] - df['volante(g)'].shift()) > 6

    # Sum all interaction changes for each row
    df['interactions_count'] = df[['changes_travao', 'changes_acelerador', 'gear_changes', 'volante_changes']].sum(axis=1).astype(int)

    # Initialize interval_interactions and start_interval_time
    interval_interactions = 0
    start_interval_time = df['tempo(s)'].iloc[0]

    # Vectorize the interval interactions calculation
    interval_interactions_list = []
    for index, current_time in enumerate(df['tempo(s)']):
        if current_time >= start_interval_time + 60:
            interval_interactions = 0
            start_interval_time = current_time

        interval_interactions += df['interactions_count'].iloc[index]
        interval_interactions_list.append(interval_interactions)

    df['interval_cumulative_interactions'] = interval_interactions_list

    # Drop temporary columns
    df.drop(columns=['changes_travao', 'changes_acelerador', 'gear_changes', 'volante_changes', 'interactions_count'], inplace=True)

    return df

def fix_negative_velocities(df):
    # Ensure all values in 'vel (km/h)' are at least 0
    df['vel (km/h)'] = df['vel (km/h)'].clip(lower=0)
    return df


def pisa_linha(df):
    df['eixo_offset(m)'] = pd.to_numeric(df['eixo_offset(m)'], errors='coerce')

    # Compute a boolean series where the condition is met
    condition_met = (4.25 <= df['eixo_offset(m)']) & (df['eixo_offset(m)'] <= 5.85)
    condition_met_2 = df['eixo_offset(m)'] >= 7.2

    combined_conditions_met = condition_met | condition_met_2

    # Create a shifted version of this series to compare each element with the previous one
    previous_condition_met = combined_conditions_met.shift(1, fill_value=False)

    # Calculate when the condition transitions from False to True
    transition_points = combined_conditions_met & ~previous_condition_met

    # Sum these transitions cumulatively to get the 'pisa_linha_count'
    df['pisa_linha_count'] = transition_points.cumsum()
    return df

# def muda_faixa(df):
#     # Compute a boolean series where the condition is met
#     condition_met = (2.1 <= df['eixo_offset(m)']) & (df['eixo_offset(m)'] <= 4.25)

#     # Create a shifted version of this series to compare each element with the previous one
#     previous_condition_met = condition_met.shift(1, fill_value=False)

#     # Calculate when the condition transitions from False to True
#     transition_points = condition_met & ~previous_condition_met

#     # Sum these transitions cumulatively to get the 'pisa_linha_count'
#     df['muda_faixa_count'] = transition_points.cumsum()

#     return df

def muda_faixa(df):
    # Assuming 'timestamp' is in seconds and sorted
    interval_size = 60  # Interval size in seconds
    
    # Compute a boolean series where the condition is met
    condition_met = (2.1 <= df['eixo_offset(m)']) & (df['eixo_offset(m)'] <= 4.25)
    
    # Create a shifted version of this series to compare each element with the previous one
    previous_condition_met = condition_met.shift(1, fill_value=False)
    
    # Calculate when the condition transitions from False to True
    transition_points = condition_met & ~previous_condition_met
    
    # Initialize the next reset time
    next_reset_time = df['tempo(s)'].iloc[0] + interval_size
    
    # Initialize the cumulative sum variable
    cum_sum = 0
    muda_faixa_counts = []

    for i in range(len(df)):
        # Check if the current timestamp exceeds the reset time
        if df['tempo(s)'].iloc[i] >= next_reset_time:
            cum_sum = 0  # Reset cumulative sum
            next_reset_time += interval_size  # Set the next reset time

        # Add transition point to cumulative sum
        if transition_points.iloc[i]:
            cum_sum += 1
        
        muda_faixa_counts.append(cum_sum)

    # Assign the calculated counts back to the DataFrame
    df['muda_faixa_count'] = muda_faixa_counts

    return df

# Function definition
def pisa_linha_reset(df):
    interval_size = 60
    # Initialize counter for conditions
    counter = 0
    counters = []
    next_reset_time = interval_size
    df['eixo_offset(m)'] = pd.to_numeric(df['eixo_offset(m)'], errors='coerce')
    # Iterate over each timestamp
    for i in range(len(df)):
        timestamp = df.loc[i, 'tempo(s)']
        eixo_offset = df.loc[i, 'eixo_offset(m)']

        # Check conditions for the current timestamp
        if (4.25 <= eixo_offset <= 5.85) or (eixo_offset >= 7.2):
            counter += 1
        
        # Add current counter value to the list
        counters.append(counter)
        
        # Check if we need to reset the counter
        if timestamp >= next_reset_time:
            counter = 0
            next_reset_time += interval_size

    # Add counters as a new feature to the DataFrame
    df['pisa_linha_count'] = counters
    
    return df

def add_metadata(df, file_path):
    # Load metadata from a .sav file

    
    metadata, _ = pyreadstat.read_sav(file_path[0])
    
    # Convert metadata to a Pandas DataFrame
    metadata_df = pd.DataFrame(metadata)
    df['Participant'] = df['Participant'].astype(int)
    # Find the row in the metadata where 'ID' matches 'Participant' in df
    # Assuming 'Participant' contains unique identifiers that match 'ID' in metadata
    participant_id = int(df['Participant'].iloc[0])  # Gets the first participant ID from df
    metadata_row = metadata_df[metadata_df['ID'] == participant_id]    
    # If no matching ID found, return the original df
    if metadata_row.empty:
        print("No matching metadata found for the given ID.")
        return df

    # Attempt to convert all columns to numeric, coercing errors to NaN
    for column in metadata_row.columns:
        metadata_row[column] = pd.to_numeric(metadata_row[column], errors='coerce')
    
    # Merge the identified metadata row with the original df
    # Assuming you want to merge on columns that exist in both dataframes
    merged_df = df.merge(metadata_row, left_on='Participant', right_on='ID', how='left')
    # Remove columns where all values are NaN

    merged_df = merged_df.dropna(axis=1, how='all')
    return merged_df



def extract_sim_sm_eyet(experiment):
    base_path = select_base_path()
    while True:
        if experiment == 'drowsiness':
            try:
                sim_txt, sim_dict = extract_all(experiment,'simulator', None,  'txt', base_path)
                # print("No valid files found. Please select the base path again.")
                # base_path = select_base_path()
                
            except ValueError as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt, sim_dict = extract_all(experiment, 'simulator', None, 'txt', base_path)
                # print("No valid files found. Please select the base path again.")
                # base_path = select_base_path()
                continue
                
        elif experiment == 'distraction':
            try:
                sim_txt = extract_all(experiment,'simulator', None,  'txt', base_path)
                if len(sim_txt) == 0: 
                    print("No valid files found. Please select the base path again.")
                    base_path = select_base_path()
                    continue
            except ValueError as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt = extract_all(experiment, 'simulator', None, 'txt', base_path)
                # print("No valid files found. Please select the base path again.")
                # base_path = select_base_path()
                continue
        print('Eyetracker preparations...')
        list_df_ver_left, list_df_ver_right, list_df_ver_ver = extract_all(experiment, 'eyetracker', 'gaze all', 'csv', base_path)
        ver = join_vergences(list_df_ver_left, list_df_ver_right, list_df_ver_ver)
       
        print('Smartwatch preparations...')
        smartwatch = extract_all(experiment, 'smartwatch', None, 'csv', base_path)
        sim_log = extract_all(experiment, 'simulator', None, 'log', base_path)

        while True:
            user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
            if user_input.lower() == 'yes':
                print('Select the file containing the .csv with the data to synchronize the dataframes.')
                sinc_path = configuration.file_path()
                smartwatch = sinc_dataframes(smartwatch, 'smartwatch', sinc_path[0])
                ver = sinc_dataframes(ver, 'eyetracker', sinc_path[0])
                break
            elif user_input.lower() == 'no':
                break
            else:
                print("Please enter 'yes' or 'no'.")
                continue
        
        smartwatch = prepare_smartwatch(smartwatch)
        print('Joining dataframes...')
        merged_df = join_log(ver, sim_log)
        merged_df_txt = join_smartwatch_txt(merged_df, sim_txt, 'downsample')
        #remove column 'Participant_y' from merged_df_txt
        merged_df_txt = [df_merged.drop('Participant_y', axis=1) for df_merged in merged_df_txt]
        #rename column 'Participant_x' to 'Participant' in merged_df_txt
        merged_df_txt = [df_merged.rename(columns={'Participant_x': 'Participant'}) for df_merged in merged_df_txt]
        merged_full = join_smartwatch_txt(merged_df_txt, smartwatch, 'other')
        print("Do you wish to add participants' metadata? DISCLAIMER: It must be on .sav format and have a feature called ID (yes/no)")
        metadata_choice = input('>')
        if metadata_choice.lower() == 'yes':
            file_path_metadata= configuration.file_path()
            merged_full = [add_metadata(df, file_path_metadata) for df in merged_full]
        elif metadata_choice.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue
        print('Do you wish select a .json file with the features to extract? (yes/no)')
        choice_features = input('>')
        if choice_features.lower() == 'yes':
            merged_full = filter_dataframes_by_features(merged_full)
        elif choice_features.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue
        return merged_full

def extract_sim_eyetracker(experiment):
    while True:
        base_path = select_base_path()
        if experiment == 'drowsiness':
            try:
                sim_txt, sim_dict = extract_all(experiment,'simulator', None,  'txt', base_path)
            except ValueError as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt, sim_dict = extract_all(experiment, 'simulator', None, 'txt', base_path)
        elif experiment == 'distraction':
            try:
                sim_txt = extract_all(experiment,'simulator', None,  'txt', base_path)
            except (ValueError, IndexError) as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt = extract_all(experiment, 'simulator', None, 'txt', base_path)
        sim_log = extract_all(experiment, 'simulator', None, 'log', base_path)
        list_df_ver_left, list_df_ver_right, list_df_ver_ver = extract_all(experiment, 'eyetracker', 'gaze all', 'csv', base_path)
        ver = join_vergences(list_df_ver_left, list_df_ver_right, list_df_ver_ver)
        while True:
            user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
            if user_input.lower() == 'yes':
                print('Select the file containing the .csv with the data to synchronize the dataframes.')
                sinc_path = configuration.file_path()
                ver = sinc_dataframes(ver, 'eyetracker', sinc_path[0])
                break
            elif user_input.lower() == 'no':
                break
            else:
                print("Please enter 'yes' or 'no'.")
                continue
        merged_df = join_log(ver, sim_log)
        merged_df_txt = join_smartwatch_txt(merged_df, sim_txt, 'downsample')
        #remove column 'Participant_y' from merged_df_txt
        merged_df_txt = [df_merged.drop('Participant_y', axis=1) for df_merged in merged_df_txt]
        #rename column 'Participant_x' to 'Participant' in merged_df_txt
        merged_df_txt = [df_merged.rename(columns={'Participant_x': 'Participant'}) for df_merged in merged_df_txt]
        print("Do you wish to add participants' metadata? DISCLAIMER: It must be on .sav format and have a feature called ID (yes/no)")
        metadata_choice = input('>')
        if metadata_choice.lower() == 'yes':
            file_path_metadata= configuration.file_path()
            merged_df_txt = [add_metadata(df, file_path_metadata) for df in merged_df_txt]
        elif metadata_choice.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue
        print('Do you wish select a .json file with the features to extract? (yes/no)')
        choice_features = input('>')
        if choice_features.lower() == 'yes':
            merged_df_txt = filter_dataframes_by_features(merged_df_txt)
        elif choice_features.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue
        return merged_df_txt

def extract_simulator(experiment): 
    base_path = select_base_path()
    while True:
        if experiment == 'distraction':
            try:
                sim_txt = extract_all(experiment,'simulator', None,  'txt', base_path)
            except ValueError as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt = extract_all(experiment, 'simulator', None, 'txt', base_path)
        elif experiment == 'drowsiness':
            try:
                sim_txt, sim_dict = extract_all(experiment,'simulator', None,  'txt', base_path)
            except ValueError as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt, sim_dict = extract_all(experiment, 'simulator', None, 'txt', base_path)
        sim_log = extract_all(experiment, 'simulator', None, 'log', base_path)
        merged_sims = merge_sims(sim_txt, sim_log)
        print("Do you wish to add participants' metadata? DISCLAIMER: It must be on .sav format and have a feature called ID (yes/no)")
        metadata_choice = input('>')
        if metadata_choice.lower() == 'yes':
            file_path_metadata= configuration.file_path()
            merged_sims = [add_metadata(df, file_path_metadata) for df in merged_sims]
        elif metadata_choice.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue
        print('Do you wish select a .json file with the features to extract? (yes/no)')
        choice_features = input('>')
        if choice_features.lower() == 'yes':
            merged_sims = filter_dataframes_by_features(merged_sims)
        elif choice_features.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue
        return merged_sims


def extract_sim_smartwatch(experiment):
    base_path = select_base_path()
    while True: 
        if experiment == 'drowsiness':
            try:
                sim_txt, sim_dict = extract_all(experiment,'simulator', None,  'txt', base_path)
            except ValueError as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt, sim_dict = extract_all(experiment, 'simulator', None, 'txt', base_path)
        elif experiment == 'distraction':
            try:
                sim_txt = extract_all(experiment,'simulator', None,  'txt', base_path)
                if len(sim_txt) == 0:
                    print("No valid files found. Please select the base path again.")
                    base_path = select_base_path()
                    continue
            except ValueError as e:
                print("Error: not enough values to unpack. You probably selected the wrong file. Please try again.")
                # Optionally, you can prompt the user to select the base path again
                base_path = select_base_path()
                sim_txt = extract_all(experiment, 'simulator', None, 'txt', base_path)
                #if sim_txt is empty prompt to select base_path again
                if len(sim_txt) == 0:
                    base_path = select_base_path()
                    sim_txt = extract_all(experiment, 'simulator', None, 'txt', base_path)

        sim_log = extract_all(experiment, 'simulator', None, 'log', base_path)
        merged_sims = merge_sims(sim_txt, sim_log)
        smartwatch = extract_all(experiment, 'smartwatch', None, 'csv', base_path)
        while True:
            user_input = input('Would you like to synchronize the dataframes? (yes/no)\n>')
            if user_input.lower() == 'yes':
                print('Select the file containing the .csv with the data to synchronize the dataframes.')
                sinc_path = configuration.file_path()
                smartwatch = sinc_dataframes(smartwatch, 'smartwatch', sinc_path[0])        
                break
            elif user_input.lower() == 'no':
                break
            else:
                print("Please enter 'yes' or 'no'.")
                continue
        
        smartwatch = prepare_smartwatch(smartwatch)
        merged_sim_sw = join_sim_w_smartwatch(merged_sims, smartwatch)
        print("Do you wish to add participants' metadata? DISCLAIMER: It must be on .sav format and have a feature called ID (yes/no)")
        metadata_choice = input('>')
        if metadata_choice.lower() == 'yes':
            file_path_metadata= configuration.file_path()
            merged_sim_sw = [add_metadata(df, file_path_metadata) for df in merged_sim_sw]
        elif metadata_choice.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue
        print('Do you wish select a .json file with the features to extract? (yes/no)')
        choice_features = input('>')
        if choice_features.lower() == 'yes':
            merged_sim_sw = filter_dataframes_by_features(merged_sim_sw)
        elif choice_features.lower() == 'no':
            pass
        else:
            print('Please enter yes or no.')
            continue

        return merged_sim_sw







    