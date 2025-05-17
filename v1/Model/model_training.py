from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List

def train_model(X_train, y_train, n_estimators=100, random_state=42, n_jobs=-1):
    model = RandomForestRegressor(n_estimators=n_estimators,random_state=random_state,n_jobs=n_jobs)
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")

    return model

def cross_validate_model(model, X_train, y_train, cv=5):

    # Perform cross-validation and calculate scores
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')

    # Convert negative MSE to positive MSE
    mse_scores = -scores
    mean_mse = np.mean(mse_scores)
    mean_r2 = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='r2'))

    print(f"Cross-validation MSE scores: {mse_scores}")
    print(f"Mean MSE: {mean_mse}")
    print(f"Mean R²: {mean_r2}")
    
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

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates the model on test data and returns MSE and R² score."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation on Test Set:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Create and save regression plots
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Regression: True vs Predicted Values')
    save_plot(plt, 'regression_scatter.png', '1')

    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    save_plot(plt, 'residuals_plot.png', '1')

    return mse, r2

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
    save_plot(plt, 'feature_importances.png', '1')

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename} successfully!")