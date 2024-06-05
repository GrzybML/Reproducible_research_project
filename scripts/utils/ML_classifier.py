import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
import shap
import os

class MLClassifier:
    def __init__(self, data, target, output_dir='output'):
        self.data = data
        self.target = target
        self.models = {
            'Logistic Regression': LogisticRegression(solver='saga'),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.param_grids = {
            'Logistic Regression': {
                'clf__C': [0.01, 0.1, 1, 10, 100]
            },
            'Random Forest': {
                'clf__n_estimators': [10, 100, 1000]
            },
            'XGBoost': {
                'clf__n_estimators': [10, 100, 1000],
                'clf__learning_rate': [0.001, 0.01, 0.1]
            }
        }
        self.best_params = {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def preprocess_data(self, data, drop_time_diff=True):
        if drop_time_diff:
            X = data.drop(columns=[self.target, 'time_diff'])
        else:
            X = data.drop(columns=[self.target])
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_models(self):
        X_train, X_test, y_train, y_test = self.preprocess_data(self.data)
        
        for model_name, model in self.models.items():
            smote = SMOTE(random_state=42, k_neighbors=5)
            pipeline = imbpipeline(steps=[('smote', smote), ('clf', model)])
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grids[model_name],
                scoring=make_scorer(roc_auc_score),
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_params[model_name] = grid_search.best_params_
            best_model = grid_search.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"AUC-ROC for {model_name}: {auc:.4f}")
            print("-----")
    
    def sensitivity_analysis(self):
        hops = [-5,-4,-3,-2,-1,0,1,2,3,4,5]  
        time_diffs = np.arange(0, 7.5, 0.5)  
        features = ['speed', 'volume', 'occupancy']
        
        results = []
        for time_diff in time_diffs:  
            filtered_data = self.data[np.isclose(self.data['time_diff'], time_diff, atol=0.1)]
            
            if filtered_data.empty:
                continue

            for hop in hops:
                for feature in features:
                    feature_name = f"{feature} (i{hop:+})"
                    if feature_name not in filtered_data.columns:
                        continue  

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
            smote = SMOTE(random_state=42, k_neighbors=5)
            pipeline = imbpipeline(steps=[('smote', smote), ('clf', model)])
            pipeline.set_params(**self.best_params[model_name])
                
            pipeline.fit(X_train[[feature_name]], y_train)
            y_pred = pipeline.predict(X_test[[feature_name]])
            y_pred_proba = pipeline.predict_proba(X_test[[feature_name]])[:, 1]
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            dr = recall_score(y_test, y_pred)
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auc_pr = auc(recall, precision)

            model_results[model_name] = {'DR': dr, 'FAR': far, 'AUC-ROC': auc_roc, 'AUC-PR': auc_pr}

        return model_results

    def generate_heatmap(self, results):
        settings = [1, 2]
        
        metrics = set()
        for res in results:
            for key in res.keys():
                if key not in {'time_diff', 'hop', 'feature', 'model'}:
                    metrics.add(key)
    
        for model_name in self.models.keys():
            for setting in settings:
                for metric in metrics:
                    df = pd.DataFrame([res for res in results if res['model'] == model_name])
            
                    if setting == 1:
                        df = df[(df['hop'] <= 0)]
                    elif setting == 2:
                        df['hop'] = [(f'+/-{abs(hop)}' if hop != 0 else '0') for hop in df['hop']]

                    df_agg = df.groupby(['hop', 'time_diff'])[metric].mean().unstack()
            
                    if setting == 2:
                        order = [f"+/-{i}" for i in range(5, 0, -1)] + ["0"]
                        df_agg = df_agg.reindex(order)

                    plt.figure(figsize=(12, 10))
                    heatmap = sns.heatmap(df_agg, annot=True, cmap='YlOrRd', fmt=".2f", cbar_kws={'label': metric, 'location': 'left'})
                    plt.title(f'Heatmap of {metric} for {model_name} (Setting {setting})')
                    heatmap.set_xlabel(None)
                    heatmap.set_ylabel(None)
                    x_labels = [f"{label} min" for label in df_agg.columns]
                    heatmap.set_xticklabels(x_labels) 
                    y_labels = [f"{label} hop" for label in df_agg.index]
                    heatmap.set_yticklabels(y_labels) 
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=90)
            
                    output_path = os.path.join(self.output_dir, f'heatmap_{model_name}_{metric}_setting_{setting}.png')
                    plt.savefig(output_path)
                    plt.close()

    def generate_summary_table(self, results):
        summary = {'Setting': [], 'Model': [], 'DR': [], 'FAR': [], 'AUC-ROC': [], 'AUC-PR': []}
        for setting in [1, 2]:
            for model_name in self.models.keys():
                if setting == 1:
                    setting_results = [res for res in results if res['model'] == model_name and res['hop'] <= 0]
                elif setting == 2:
                    setting_results = [res for res in results if res['model'] == model_name]
                    for res in setting_results:
                        res['hop'] = f'+/-{abs(res["hop"])}' if res['hop'] != 0 else '0'

                dr_values = [res['DR'] for res in setting_results]
                far_values = [res['FAR'] for res in setting_results]
                auc_roc_values = [res['AUC-ROC'] for res in setting_results]
                auc_pr_values = [res.get('AUC-PR', np.nan) for res in setting_results] 

                summary['Setting'].append(f'Setting {setting}')
                summary['Model'].append(model_name)
                summary['DR'].append(f"{np.mean(dr_values):.2f} ({np.std(dr_values):.2f})/{np.median(dr_values):.2f}")
                summary['FAR'].append(f"{np.mean(far_values):.2f} ({np.std(far_values):.2f})/{np.median(far_values):.2f}")
                summary['AUC-ROC'].append(f"{np.mean(auc_roc_values):.2f} ({np.std(auc_roc_values):.2f})/{np.median(auc_roc_values):.2f}")
                summary['AUC-PR'].append(f"{np.nanmean(auc_pr_values):.2f} ({np.nanstd(auc_pr_values):.2f})/{np.nanmedian(auc_pr_values):.2f}")
        
        df_summary = pd.DataFrame(summary)
        return df_summary    
    
    def plot_shap_values(self, results):
        best_auc_pr = 0
        best_model_details = None
        
        for setting in [1, 2]:
            for model_name in self.models.keys():
                df_results = pd.DataFrame([res for res in results if res['model'] == model_name])
                if setting == 1:
                    setting_results = df_results[df_results['hop'] <= 0]
                elif setting == 2:
                    df_results['hop'] = [f'+/-{abs(hop)}' if hop != 0 else '0' for hop in df_results['hop']]
                    setting_results = df_results
                    
                if not setting_results.empty:
                    df_agg = setting_results.groupby(['hop', 'time_diff'])['AUC-PR'].mean().unstack()
                    max_auc_pr_index = df_agg.stack().idxmax()  
                    max_auc_pr = df_agg.loc[max_auc_pr_index] 

                    if max_auc_pr > best_auc_pr:  
                        best_auc_pr = max_auc_pr
                        best_model_details = {
                            'time_diff': max_auc_pr_index[1],
                            'hop': max_auc_pr_index[0],
                            'model_name': model_name,
                            'setting': setting
                        }
        
        best_data = self.data[(self.data['time_diff'] == best_model_details['time_diff'])]
        X_train, _, y_train, _ = self.preprocess_data(best_data, drop_time_diff=True)
        
        bool_columns = X_train.select_dtypes(include=['bool']).columns
        X_train[bool_columns] = X_train[bool_columns].astype(int)
        
        best_model = self.models[best_model_details['model_name']]
        pipeline = imbpipeline(steps=[('smote', SMOTE(random_state=42)), ('clf', best_model)])
        pipeline.set_params(**self.best_params[best_model_details['model_name']])
        pipeline.fit(X_train, y_train)

        explainer = shap.Explainer(pipeline.named_steps['clf'], X_train)
        shap_values = explainer(X_train)

        print(f"SHAP values for {best_model_details['model_name']} at TTDA {best_model_details['time_diff']} min under Setting {best_model_details['setting']}")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values[...,1], X_train, plot_type="dot")
        plt.show()