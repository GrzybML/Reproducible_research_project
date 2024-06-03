import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, precision_score, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
import shap

class MLClassifier:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.models = {
            'Logistic Regression': LogisticRegression(solver='saga'),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.param_grids = {
            'Logistic Regression': {
                'classifier__C': [0.01, 0.1, 1, 10, 100]
            },
            'Random Forest': {
                'classifier__n_estimators': [10, 100, 1000]
            },
            'XGBoost': {
                'classifier__n_estimators': [10, 100, 1000],
                'classifier__learning_rate': [0.001, 0.01, 0.1]
            }
        }
        self.best_params = {}
    
    def preprocess_data(self, data, drop_time_diff=True):
        if drop_time_diff:
            X = data.drop(columns=[self.target, 'time_diff'])
        else:
            X = data.drop(columns=[self.target])
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def calculate_metrics_for_combinations(self):
        hops = [-5,-4,-3,-2,-1,0,1,2,3,4,5]  # From i-5 to i+5
        time_diffs = np.arange(-4, 7.5, 0.5)  # Correct use of time_diffs
        features = ['speed', 'volume', 'occupancy']
        
        results = []
        for time_diff in time_diffs:  # Use predefined time_diffs
            filtered_data = self.data[np.isclose(self.data['time_diff'], time_diff, atol=0.1)]
            
            if filtered_data.empty:
                continue

            for hop in hops:
                for feature in features:
                    feature_name = f"{feature} (i{hop:+})"
                    if feature_name not in filtered_data.columns:
                        continue  # Skip if the feature does not exist in the data

                    X_train, X_test, y_train, y_test = self.preprocess_data(filtered_data, drop_time_diff=False)
                    model_results = self.train_models_specific_feature(X_train, X_test, y_train, y_test, feature_name)
                    
                    for model_name, metrics in model_results.items():
                        results.append({
                            'time_diff': time_diff, 
                            'hop': hop, 
                            'feature': feature, 
                            'model': model_name, 
                            **metrics
                        })
                    
        return results

    def train_models_specific_feature(self, X_train, X_test, y_train, y_test, feature_name):
        model_results = {}
        for model_name, model in self.models.items():
            model.fit(X_train[[feature_name]], y_train)  # Train using only the specific feature
            y_pred = model.predict(X_test[[feature_name]])
            y_pred_proba = model.predict_proba(X_test[[feature_name]])[:, 1]
            
            dr = recall_score(y_test, y_pred)
            far = 1 - precision_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auc_pr = auc(recall, precision)

            model_results[model_name] = {'DR': dr, 'FAR': far, 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr}

        return model_results

    def generate_heatmap(self, results, metric, setting):
        for model_name in self.models.keys():
            df = pd.DataFrame([res for res in results if res['model'] == model_name])
            
            if setting == 1:
                df = df[(df['hop'] >= 0) & (df['time_diff'] >= 0)]
            elif setting == 2:
                df = df[(df['hop'] <= 0) & (df['time_diff'] >= 0)]

            #print("Filtered DataFrame for heatmap:")
            #print(df['hop'].unique())
            
            # Zmiana kierunku agregacji danych - 'hop' staje się kolumnami
            df_agg = df.groupby(['hop', 'time_diff'])[metric].mean().unstack()

            plt.figure(figsize=(12, 10))
            # Utworzenie heatmapy z zamienionymi osiami
            sns.heatmap(df_agg, annot=True, cmap='YlOrRd', fmt=".2f", cbar_kws={'label': metric})
            plt.title(f'Heatmap of {metric} for {model_name} (Setting {setting})')
            plt.ylabel('Hop')  # Zmieniono etykietę osi X
            plt.xlabel('Time Diff (min)')  # Zmieniono etykietę osi Y
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.show()

    def plot_shap_values(self, model_name='XGBoost'):
        # Use the entire dataset for SHAP value analysis
        X_train, X_test, y_train, y_test = self.preprocess_data(self.data, drop_time_diff=False)
        model = self.models[model_name]
        model.fit(X_train, y_train)

        # Create the SHAP explainer and calculate SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # Plot SHAP values using SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, plot_type="dot")
        plt.show()
    
    def generate_summary_table(self, results):
        summary = {'Setting': [], 'Model': [], 'DR': [], 'FAR': [], 'AUC-ROC': [], 'AUC-PR': []}
        for setting in [1, 2]:
            for model_name in self.models.keys():
                if setting == 1:
                    setting_results = [res for res in results if res['model'] == model_name and res['hop'] >= 0 and res['time_diff'] >= 0]
                elif setting == 2:
                    setting_results = [res for res in results if res['model'] == model_name and res['hop'] <= 0 and res['time_diff'] >= 0]

                dr_values = [res['DR'] for res in setting_results]
                far_values = [res['FAR'] for res in setting_results]
                auc_roc_values = [res['AUC-ROC'] for res in setting_results]
                auc_pr_values = [res.get('AUC-PR', np.nan) for res in setting_results]  # Use .get() to avoid KeyError

                summary['Setting'].append(f'Setting {setting}')
                summary['Model'].append(model_name)
                summary['DR'].append(f"{np.mean(dr_values):.2f} ({np.std(dr_values):.2f})/{np.median(dr_values):.2f}")
                summary['FAR'].append(f"{np.mean(far_values):.2f} ({np.std(far_values):.2f})/{np.median(far_values):.2f}")
                summary['AUC-ROC'].append(f"{np.mean(auc_roc_values):.2f} ({np.std(auc_roc_values):.2f})/{np.median(auc_roc_values):.2f}")
                summary['AUC-PR'].append(f"{np.nanmean(auc_pr_values):.2f} ({np.nanstd(auc_pr_values):.2f})/{np.nanmedian(auc_pr_values):.2f}")
        
        df_summary = pd.DataFrame(summary)
        return df_summary

# Usage
# Assuming 'data' is a DataFrame loaded with your data and 'target' is the target column name

