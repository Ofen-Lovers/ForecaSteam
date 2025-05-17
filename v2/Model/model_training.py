from sklearn.ensemble import RandomForestRegressor #, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score #, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint

# For LightGBM (optional)
# import lightgbm as lgb

def tune_random_forest_hyperparameters(X_train, y_train, n_iter=20, cv=3, random_state=42):
    print("\nTuning RandomForestRegressor hyperparameters...")
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7] # Replaced 'auto'
    }
    
    # Pass random_state to RF for reproducibility of tuning itself, but it's not a tuned param
    rf_model = RandomForestRegressor(random_state=random_state, n_jobs=-1) 
    
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=n_iter,  # Number of parameter settings that are sampled
        cv=cv,          # Cross-validation folds for tuning
        scoring='neg_mean_squared_error', # For regression
        random_state=random_state, # For reproducibility of RandomizedSearchCV's sampling
        n_jobs=-1,      # Use all available cores
        verbose=1       # Show progress
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best hyperparameters found: {random_search.best_params_}")
    print(f"Best CV MSE: {-random_search.best_score_:.4f}") # MSE is negative in scoring
    
    # The best_params_ from RandomizedSearchCV will not include random_state 
    # unless random_state was part of the param_distributions, which it isn't here.
    return random_search.best_params_

def train_model(X_train, y_train, model_type='random_forest', params=None, random_state=42):
    if params is None:
        params_copy = {} # Work with a copy to avoid modifying the original dict if passed
    else:
        params_copy = params.copy()


    if model_type == 'random_forest':
        # Ensure default n_jobs=-1 if not in params
        if 'n_jobs' not in params_copy:
             params_copy['n_jobs'] = -1
        
        # Remove 'random_state' from params_copy if it exists, as it's passed directly
        # This prevents the TypeError. The direct random_state argument takes precedence.
        if 'random_state' in params_copy:
            del params_copy['random_state']
            
        model = RandomForestRegressor(random_state=random_state, **params_copy)
    # Example for LightGBM (can be uncommented and expanded)
    # elif model_type == 'lightgbm':
    #     if 'n_jobs' not in params_copy:
    #          params_copy['n_jobs'] = -1
    #     if 'random_state' in params_copy: # CatBoost/LGBM might use 'random_seed' or similar
    #         del params_copy['random_state'] 
    #     model = lgb.LGBMRegressor(random_state=random_state, **params_copy) # Or 'random_seed'
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    # Use params (original) for printing, as params_copy might have had random_state removed
    print(f"\n{model_type.replace('_', ' ').title()} model trained successfully with parameters: {params if params else 'defaults'}.")
    return model

def cross_validate_model(model, X_train, y_train, cv=5, is_classifier=False):
    if is_classifier:
        # Example for classifier
        # mse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy') # or 'f1_macro' etc.
        # mean_mse = np.mean(mse_scores)
        # print(f"Cross-validation Accuracy scores: {mse_scores}")
        # print(f"Mean Accuracy: {mean_mse}")
        # # R2 is not applicable for classification in the same way
        # mean_r2 = np.nan 
        pass # Not implemented for classifier yet
    else: # Regression
        neg_mse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        mse_scores = -neg_mse_scores
        mean_mse = np.mean(mse_scores)
        
        r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        mean_r2 = np.mean(r2_scores)

        print(f"\nCross-validation MSE scores: {mse_scores}")
        print(f"Mean CV MSE: {mean_mse:.4f}")
        print(f"Mean CV R²: {mean_r2:.4f}")
    
    return mean_mse, mean_r2

def evaluate_model(model, X_test, y_test, is_classifier=False):
    y_pred = model.predict(X_test)
    
    if is_classifier:
        # acc = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred, average='macro') # or 'weighted'
        # print(f"Accuracy: {acc:.4f}")
        # print(f"F1 Score (Macro): {f1:.4f}")
        # return acc, f1 # Or other relevant classification metrics
        pass # Not implemented for classifier yet
        return None, None # Placeholder
    else: # Regression
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Evaluation on Test Set:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R² Score: {r2:.4f}") 
        return mse, r2

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename} successfully!")

def print_feature_importances(model, feature_names, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print(f"\nTop {top_n} Feature Importances:")
        print(feature_importance_df.head(top_n))
    else:
        print("\nModel does not have feature_importances_ attribute (e.g., non-tree-based model or not fitted).")