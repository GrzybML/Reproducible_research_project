import os
import pandas as pd

# Load data
accident_path = 'data/input/bestData/accident'
nonaccident_path = 'data/input/bestData/non-accident'

all_accident = os.listdir(accident_path)
all_nonaccident = os.listdir(nonaccident_path)

accidentI24_files = [file for file in all_accident if 'I24' in file]
accidentI75_files = [file for file in all_accident if 'I75' in file]

nonaccidentI24_files = [file for file in all_nonaccident if 'I24' in file]
nonaccidentI75_files = [file for file in all_nonaccident if 'I75' in file]

# Function to read CSV files
def read_csv_files(file_list, folder_path):
    data_frames = []
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame()
    
# Read I24 accident data
accidentI24_data = read_csv_files(accidentI24_files, accident_path)
filter_condition = accidentI24_data['type'].isin(["Suspected Minor Injury", "Suspected Serious Injury", "Fatal"])
accidentI24_data = accidentI24_data[filter_condition]
accidentI24_data

# Read I24 non-accident data
nonaccidentI24_data = read_csv_files(nonaccidentI24_files, nonaccident_path)
nonaccidentI24_data

# Read I75 accident data
accidentI75_data = read_csv_files(accidentI75_files, accident_path)
filter_condition = accidentI75_data['type'].isin(["Suspected Minor Injury", "Suspected Serious Injury", "Fatal"])
accidentI75_data = accidentI75_data[filter_condition]
accidentI75_data

# Read I75 non-accident data
nonaccidentI75_data = read_csv_files(nonaccidentI75_files, nonaccident_path)
nonaccidentI75_data

# Merge I24 data and write to CSV
I24_data = pd.concat([accidentI24_data, nonaccidentI24_data], ignore_index=True)
I24_data
I24_data.to_csv('data/input/I24_data.csv', index=False)

# Merge I75 data and write to CSV
I75_data = pd.concat([accidentI75_data, nonaccidentI75_data], ignore_index=True)
I75_data
I75_data.to_csv('data/input/I75_data.csv', index=False)