import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For potentially nicer plots
import joblib

def load():
    # Load trained model and the final feature list used for training
    try:
        rf_model = joblib.load('ForecaSteam.pkl')
        final_feature_columns = joblib.load('final_feature_columns.pkl')
        # Ensure you load the model trained WITH post-launch features if that's the one you want to interpret
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Make sure model and feature list are saved.")
        # Handle error appropriately

    return rf_model, final_feature_columns

def feature_importance(rf_model, final_feature_columns):
    # Get feature importances
    importances = rf_model.feature_importances_

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': final_feature_columns,
        'Importance': importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("Top 20 Feature Importances (Mean Decrease in Impurity):")
    print(feature_importance_df.head(20))

    # Visualize the top N features
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20]) # Plot top 20
    plt.xlabel("Importance (Mean Decrease in Impurity)")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances from Random Forest")
    plt.gca().invert_yaxis() # Display most important at the top
    plt.tight_layout()
    plt.show()

def main():
    rf_model, final_feature_columns = load()

    feature_importance(rf_model, final_feature_columns)