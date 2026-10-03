from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def train_and_tune_rf_sklearn(X_train, y_train, random_state=42, n_iter=10, cv=3):
    """
    Tunes Random Forest Classifier hyperparameters using RandomizedSearchCV.
    
    Param Grid:
    - n_estimators: [50, 100, 200, 300]
    - max_depth: [10, 30, 50, 110, None]
    - max_features: ['sqrt', 'log2', 4]
    - min_samples_split: [2, 5, 10]
    """
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 30, 50, 110, None],
        'max_features': ['sqrt', 'log2', 4],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='recall', # Optimize for recall as stated in paper Section 9.1
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    print("Best RF Hyperparameters:", search.best_params_)
    print("Best CV Recall Score:", search.best_score_)
    
    return search.best_estimator_, search.best_params_
