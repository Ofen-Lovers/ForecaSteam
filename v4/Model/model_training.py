from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # ADDED RandomForestRegressor
from sklearn.metrics import ( # EXPANDED this import
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score # ADDED regression metrics
)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, List, Union

def tune_random_forest_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                                       is_classifier: bool = True,
                                       n_iter: int = 20, cv: int = 3,
                                       random_state: int = 42) -> dict:
    """
    Tunes RandomForest hyperparameters using RandomizedSearchCV.
    Can be used for both Classifier and Regressor.
    """
    model_name = "RandomForestClassifier" if is_classifier else "RandomForestRegressor"
    print(f"\nTuning {model_name} hyperparameters...")
    
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'class_weight': [None, 'balanced', 'balanced_subsample'] if is_classifier else [None] 
    }
    if not is_classifier:
        param_dist_reg = param_dist.copy()
        if 'class_weight' in param_dist_reg: # Ensure it's removed if present
            del param_dist_reg['class_weight']
        param_dist_to_use = param_dist_reg
    else:
        param_dist_to_use = param_dist


    if is_classifier:
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        scoring_metric = 'f1_weighted' 
    else: 
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1) # Now defined
        scoring_metric = 'neg_mean_squared_error'
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist_to_use,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring_metric,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    print(f"Best hyperparameters found for {model_name}: {best_params}")
    if is_classifier:
        print(f"Best CV {scoring_metric}: {random_search.best_score_:.4f}")
    else:
        print(f"Best CV Score (negative, so higher is better for MSE): {random_search.best_score_:.4f}") 
        print(f"Corresponding Best CV MSE: {-random_search.best_score_:.4f}")
    
    return best_params

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                model_type: str = 'random_forest_classifier', 
                params: dict = None, random_state: int = 42):
    """
    Trains a machine learning model.
    """
    params_copy = params.copy() if params else {}

    if model_type == 'random_forest_classifier':
        if 'n_jobs' not in params_copy:
             params_copy['n_jobs'] = -1
        if 'random_state' in params_copy:
            del params_copy['random_state']
        # Ensure class_weight is handled if present in params_copy for classifier
        if 'class_weight' not in params_copy and 'class_weight' in (params or {}): # If it was in original params
            params_copy['class_weight'] = params['class_weight']

        model = RandomForestClassifier(random_state=random_state, **params_copy)

    elif model_type == 'random_forest_regressor': 
        if 'n_jobs' not in params_copy:
             params_copy['n_jobs'] = -1
        if 'random_state' in params_copy:
            del params_copy['random_state']
        # Ensure 'class_weight' is NOT passed to Regressor
        if 'class_weight' in params_copy:
            del params_copy['class_weight']
        model = RandomForestRegressor(random_state=random_state, **params_copy) # Now defined
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)
    print(f"\n{model_type.replace('_', ' ').title()} model trained successfully with parameters: {params_copy if params_copy else 'defaults'}.")
    return model

def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                         is_classifier: bool = True, 
                         cv: int = 5) -> tuple: 
    """
    Performs cross-validation on the training set.
    """
    if is_classifier:
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        mean_cv_f1 = np.mean(cv_f1_scores)
        
        cv_accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_cv_accuracy = np.mean(cv_accuracy_scores)

        print(f"\nCross-validation F1 (weighted) scores: {cv_f1_scores}")
        print(f"Mean CV F1 (weighted): {mean_cv_f1:.4f}")
        print(f"Cross-validation Accuracy scores: {cv_accuracy_scores}")
        print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
        return mean_cv_f1, mean_cv_accuracy
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

def save_plot(plt, filename: str, version: str):
    """Saves a plot to the version-specific images directory."""
    # Create version-specific images directory
    images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(images_dir, filename)
    try:
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")
    finally:
        plt.close()

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   is_classifier: bool = True, 
                   label_encoder: Optional[LabelEncoder] = None):
    """
    Evaluates the model on test data and returns performance metrics.
    For classification: returns accuracy and F1 score
    For regression: returns MSE and R² score
    """
    y_pred = model.predict(X_test)
    
    if is_classifier:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nModel Evaluation on Test Set (Classification):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Weighted): {f1:.4f}")
        
        # Get class names for reporting
        report_labels = sorted(list(set(y_test)))
        report_target_names = None
        if label_encoder is not None:
            try:
                report_target_names = label_encoder.inverse_transform(report_labels)
                # Convert to list if it's a numpy array
                if isinstance(report_target_names, np.ndarray):
                    report_target_names = report_target_names.tolist()
            except:
                pass

        if report_target_names is not None and len(report_target_names) == len(report_labels):
            print(classification_report(y_test, y_pred, labels=report_labels, target_names=report_target_names, zero_division=0))
        else: # Fallback if target_names creation failed
            print(classification_report(y_test, y_pred, labels=report_labels, zero_division=0))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=report_labels)
        print(cm)
        
        # Adjust figure size dynamically based on number of classes
        fig_width = max(10, len(report_target_names or []) * 0.8)
        fig_height = max(8, len(report_target_names or []) * 0.6)
        plt.figure(figsize=(fig_width, fig_height))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=report_target_names or report_labels, 
                    yticklabels=report_target_names or report_labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save confusion matrix plot
        save_plot(plt, 'confusion_matrix.png', '4')

        return accuracy, f1
    else: # Regression
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Evaluation on Test Set (Regression):")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Create and save regression plots
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Regression: True vs Predicted Values')
        save_plot(plt, 'regression_scatter.png', '4')

        # Residuals plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        save_plot(plt, 'residuals_plot.png', '4')

        return mse, r2

def save_model(model, filename: str):
    """Saves the trained model to a file using joblib."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename} successfully!")

def print_feature_importances(model, feature_names: List[str], top_n: int = 20):
    """Prints and plots the top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature importances
    print("\nTop {} Feature Importances:".format(top_n))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    print(importance_df)
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.title('Top {} Feature Importances'.format(top_n))
    plt.tight_layout()
    save_plot(plt, 'feature_importances.png', '4')