from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

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
    
    def preprocess_data(self, drop_time_diff = True):
        if drop_time_diff:
            X = self.data.drop(columns=[self.target, 'time_diff'])
        else:
            X = self.data.drop(columns=[self.target])
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_models(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        for model_name, model in self.models.items():
            smote = SMOTE(random_state=42, k_neighbors=5)
            pipeline = imbpipeline(steps=[('smote', smote), ('classifier', model)])
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grids[model_name],
                scoring=make_scorer(roc_auc_score),
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                n_jobs=1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_params[model_name] = grid_search.best_params_
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"AUC-ROC for {model_name}: {auc:.4f}")
            print("-----")
            