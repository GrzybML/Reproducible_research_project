import pandas as pd
from scripts.utils.data_preprocessor import DataPreprocessor

# Load the data
data_path = './data/I75_data.csv'
data = pd.read_csv(data_path)

# Initialize the DataPreprocessor
preprocessor = DataPreprocessor(data)

# Perform preprocessing steps
preprocessor.replace_missings()
preprocessor.handle_missings()
preprocessor.one_hot_encode()

# Save the preprocessed data
preprocessor.save_preprocessed_data('./data/preprocessed_data.csv')

print("Data preprocessing complete and file saved to './data/preprocessed_data.csv'")
