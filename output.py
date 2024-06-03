import pandas as pd
from ML_classifier import MLClassifier

# Load the preprocessed data
data_path = '/Users/polaparol/Documents/DS-sem4/rresearch/rr_project/Reproducible_research_project/preprocessed_I24_data.csv'
preprocessed_I24_data = pd.read_csv(data_path, low_memory=False)

# Subset the data for testing or demonstration
preprocessed_I24_data_short = preprocessed_I24_data[:1100]

# Initialize the classifier with the subset data
classifier = MLClassifier(data=preprocessed_I24_data_short, target='incident at sensor (i)')
results = classifier.calculate_metrics_for_combinations()

#DR
classifier.generate_heatmap(results, 'DR', setting=1)  
classifier.generate_heatmap(results, 'DR', setting=2)  

#FAR
classifier.generate_heatmap(results, 'FAR', setting=1) 
classifier.generate_heatmap(results, 'FAR', setting=2)  

#AUC-ROC
classifier.generate_heatmap(results, 'AUC-ROC', setting=1)  
classifier.generate_heatmap(results, 'AUC-ROC', setting=2)  

#AUC-PR
classifier.generate_heatmap(results, 'AUC-PR', setting=1)  
classifier.generate_heatmap(results, 'AUC-PR', setting=2)  

#SHAP plot values
classifier.plot_shap_values(model_name='XGBoost') 

#Summary table
df_summary = classifier.generate_summary_table(results)
print(df_summary)