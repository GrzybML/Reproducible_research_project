import pandas as pd
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        
    def replace_missings(self, columns=None):
        if columns is None:
            columns = self.data.columns
        
        for col in columns:
            if self.data[col].isnull().any() and self.data[col].dtype in ['int64', 'float64']:
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            elif self.data[col].isnull().any() and self.data[col].dtype == 'object':
                self.data[col] = self.data[col].fillna('Unknown')
                
    def handle_missings(self, columns=None):
        if columns is None:
            columns = self.data.columns
            
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        imputer = SimpleImputer(strategy='constant', fill_value=-99999)
        self.data[numerical_cols] = imputer.fit_transform(self.data[numerical_cols])
    
    def one_hot_encode(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns
        
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)
    
    def get_preprocessed_data(self):
        return self.data
    
    def save_preprocessed_data(self, file_path='./data/preprocessed_data.csv'):
        self.data.to_csv(file_path, index=False)
