# %%
import pandas as pd
from data_preprocessor import DataPreprocessor
from ML_classifier import MLClassifier

# %%
data = pd.read_csv('data/preprocessed_I24_data.csv', low_memory=False)
data
# %%
data_short = data[:1000]
data_short
# %%
classifier = MLClassifier(data=data_short, target='incident at sensor (i)')
classifier.train_models()
# %%
classifier.sensitivity_analysis()
# %%
results = classifier.sensitivity_analysis()

# %%
classifier.generate_heatmap(results) 

# %%
classifier.plot_shap_values(results) 

# %%
classifier.generate_summary_table(results)
# %%
