from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint

def tune_random_forest_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, 
                                       n_iter: int = 20, cv: int = 3, 
                                       random_state: int = 42) -> dict:
    """
    Tunes RandomForestRegressor hyperparameters using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_iter (int): Number of parameter settings that are sampled.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Best hyperparameters found.
    """
    print("\nTuning RandomForestRegressor hyperparameters...")
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7] 
    }
    
    rf_model = RandomForestRegressor(random_state=random_state, n_jobs=-1) 
    
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    print(f"Best hyperparameters found: {best_params}")
    print(f"Best CV MSE: {-random_search.best_score_:.4f}")
    
    return best_params

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                model_type: str = 'random_forest', 
                params: dict = None, random_state: int = 42):
    """
    Trains a machine learning model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_type (str): Type of model to train (e.g., 'random_forest').
        params (dict, optional): Hyperparameters for the model. Defaults to None.
        random_state (int): Random seed for reproducibility.

    Returns:
        Trained model object.
    """
    params_copy = params.copy() if params else {}

    if model_type == 'random_forest':
        if 'n_jobs' not in params_copy:
             params_copy['n_jobs'] = -1
        if 'random_state' in params_copy: # Ensure direct argument takes precedence
            del params_copy['random_state']
        model = RandomForestRegressor(random_state=random_state, **params_copy)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    print(f"\n{model_type.replace('_', ' ').title()} model trained successfully with parameters: {params if params else 'defaults'}.")
    return model

def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series, 
                         cv: int = 5) -> tuple[float, float]:
    """
    Performs cross-validation on the training set.

    Args:
        model: Trained model object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        cv (int): Number of cross-validation folds.

    Returns:
        tuple[float, float]: Mean CV MSE and Mean CV R-squared.
    """
    neg_mse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    mse_scores = -neg_mse_scores
    mean_mse = np.mean(mse_scores)
    
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    mean_r2 = np.mean(r2_scores)

    print(f"\nCross-validation MSE scores: {mse_scores}")
    print(f"Mean CV MSE: {mean_mse:.4f}")
    print(f"Mean CV R²: {mean_r2:.4f}")
    
    return mean_mse, mean_r2

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
    """
    Evaluates the model on the test set.

    Args:
        model: Trained model object.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        tuple[float, float]: MSE and R-squared score.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation on Test Set:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}") 
    return mse, r2

def save_model(model, filename: str):
    """Saves the trained model to a file using joblib."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename} successfully!")

def print_feature_importances(model, feature_names: list, top_n: int = 20):
    """Prints the top N feature importances from a tree-based model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print(f"\nTop {top_n} Feature Importances:")
        print(feature_importance_df.head(top_n))
    else:
        print("\nModel does not have feature_importances_ attribute.")