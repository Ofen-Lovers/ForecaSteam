import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter # For robust stratification check

# --- Configuration ---
# Assuming this script is in the root of your ForecaSteam project directory
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(PROJECT_ROOT_DIR, 'pkl')
IMAGES_DIR = os.path.join(PROJECT_ROOT_DIR, 'images')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'Processed_Data')
TARGET_VARIABLE_NAME = 'Estimated owners'  # Make sure this matches the target in your processed data

def load_artifacts():
    """Loads the trained model, feature columns, and label encoder."""
    try:
        model = joblib.load(os.path.join(PKL_DIR, 'ForecaSteam_Classifier.pkl'))
        feature_columns = joblib.load(os.path.join(PKL_DIR, 'feature_columns.pkl'))
        label_encoder = joblib.load(os.path.join(PKL_DIR, 'label_encoder.pkl'))
        print("Model and artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}. Make sure all .pkl files are present in '{PKL_DIR}'.")
        return None, None, None
    return model, feature_columns, label_encoder

def load_and_split_processed_data(feature_columns_list):
    """Loads processed_steam.csv and splits it to get X_test, y_test."""
    processed_csv_path = os.path.join(PROCESSED_DATA_DIR, 'processed_steam.csv')
    try:
        processed_df = pd.read_csv(processed_csv_path)
        print(f"Loaded data from '{processed_csv_path}' with shape {processed_df.shape}")

        if TARGET_VARIABLE_NAME not in processed_df.columns:
            print(f"Error: Target variable '{TARGET_VARIABLE_NAME}' not found in '{processed_csv_path}'.")
            return None, None

        y_encoded = processed_df[TARGET_VARIABLE_NAME]

        missing_features = [col for col in feature_columns_list if col not in processed_df.columns]
        if missing_features:
            print(f"Warning: The following features from feature_columns.pkl are missing in {processed_csv_path}: {missing_features}")
            print("These columns will be added with 0. This might indicate an inconsistency.")
            for col in missing_features:
                processed_df[col] = 0
        
        X = processed_df[feature_columns_list]

        # Stratification check
        y_counts = Counter(y_encoded)
        min_class_count = min(y_counts.values()) if y_counts else 0
        
        # Determine if stratification is possible
        can_stratify = True
        if not y_counts or len(y_counts) <= 1 or min_class_count < 2:
            can_stratify = False
            print(f"Warning: Stratification cannot be performed. Conditions not met.")
            if not y_counts: print("Reason: Target variable is empty or counts could not be determined.")
            elif len(y_counts) <=1: print("Reason: Only one class present in target.")
            elif min_class_count < 2: print(f"Reason: Least populated class has {min_class_count} members (minimum 2 required for stratification).")
        
        stratify_option = y_encoded if can_stratify else None
        if stratify_option is None:
            print("Proceeding with non-stratified split.")
        else:
            print("Proceeding with stratified split.")


        _, X_test, _, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_option
        )
        print(f"Data split: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
        return X_test, y_test

    except FileNotFoundError:
        print(f"Error: Processed data file '{processed_csv_path}' not found.")
        return None, None
    except ValueError as ve: # Catch the specific ValueError from train_test_split
        if "The least populated class in y has only 1 member" in str(ve):
            print(f"Error during train_test_split (likely due to stratification issue): {ve}")
            print("Attempting split without stratification...")
            try:
                _, X_test, _, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=None
                )
                print(f"Data split (non-stratified): X_test shape {X_test.shape}, y_test shape {y_test.shape}")
                return X_test, y_test
            except Exception as e_nostrat:
                print(f"Error during non-stratified split: {e_nostrat}")
                return None, None
        else: # Other ValueError
            print(f"An error occurred while loading or splitting processed data: {ve}")
            return None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading or splitting processed data: {e}")
        return None, None

def save_figure(fig, filename):
    """Saves a matplotlib figure to the IMAGES_DIR."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plot_path = os.path.join(IMAGES_DIR, filename)
    try:
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot '{filename}': {e}")
    plt.close(fig)

def plot_and_save_feature_importance(model, feature_names, top_n=20):
    """Generates and saves the feature importance plot."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 20 Feature Importances (Mean Decrease in Impurity):")
    print(feature_importance_df.head(top_n))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feature_importance_df['Feature'][:top_n], feature_importance_df['Importance'][:top_n])
    ax.set_xlabel("Importance (Mean Decrease in Impurity)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importances from Random Forest")
    ax.invert_yaxis() # Display most important at the top
    plt.tight_layout()
    
    save_figure(fig, 'feature_importances.png')

def plot_and_save_confusion_matrix(y_true, y_pred, label_encoder, filename="confusion_matrix.png"):
    """Generates and saves the confusion matrix plot."""
    # Corrected: Use np.unique for both y_true (Pandas Series) and y_pred (NumPy array)
    numeric_labels = sorted(np.union1d(np.unique(y_true), np.unique(y_pred)).astype(int))
    try:
        class_names = label_encoder.inverse_transform(numeric_labels)
    except ValueError:
        print("Warning: Could not map all numeric labels to class names. Using numeric labels in plot.")
        class_names = [str(l) for l in numeric_labels]

    cm = confusion_matrix(y_true, y_pred, labels=numeric_labels)
    
    fig_width = max(10, len(class_names) * 0.8)
    fig_height = max(8, len(class_names) * 0.6)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar=True) 
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_figure(fig, filename)

def plot_and_save_classification_report(y_true, y_pred, label_encoder, filename="classification_report.png"):
    """Generates and saves the classification report as a table image."""
    # Corrected: Use np.unique for both y_true (Pandas Series) and y_pred (NumPy array)
    numeric_labels = sorted(np.union1d(np.unique(y_true), np.unique(y_pred)).astype(int))
    try:
        class_names = label_encoder.inverse_transform(numeric_labels)
    except ValueError:
        print("Warning: Could not map all numeric labels to class names for report. Using numeric labels.")
        class_names = [str(l) for l in numeric_labels]

    report_dict = classification_report(y_true, y_pred, labels=numeric_labels, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    
    print("\nClassification Report (Text):")
    print(classification_report(y_true, y_pred, labels=numeric_labels, target_names=class_names, zero_division=0))

    fig, ax = plt.subplots(figsize=(12, max(4, len(report_df) * 0.4))) 
    ax.axis('tight')
    ax.axis('off')
    
    table_data = report_df.round(3).reset_index() 
    
    the_table = ax.table(cellText=table_data.values,
                         colLabels=table_data.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2] + [0.12]*(len(table_data.columns)-1)) 

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1.1, 1.1)
    
    for (i, j), cell in the_table.get_celld().items():
        if i == 0: 
            cell.set_text_props(weight='bold')
        if j == 0 and i > 0: 
             cell.set_text_props(weight='bold')

    plt.title("Classification Report", fontsize=14, y=0.98) 
    
    save_figure(fig, filename)

def main():
    print("--- Starting Independent Data Interpretation Script ---")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    model, feature_columns, label_encoder = load_artifacts()
    if not model or not feature_columns or not label_encoder:
        print("Halting script due to failure in loading artifacts.")
        return

    print("\n--- 1. Generating Feature Importance Plot ---")
    plot_and_save_feature_importance(model, feature_columns)

    print("\n--- Loading Test Data for Evaluation Metrics ---")
    X_test, y_test_encoded = load_and_split_processed_data(feature_columns)
    
    if X_test is None or y_test_encoded is None:
        print("Failed to load test data. Skipping Classification Report and Confusion Matrix generation.")
        return

    print("\n--- Performing Predictions on Test Data ---")
    try:
        y_pred_encoded = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("This might be due to inconsistencies between loaded features and model expectations.")
        return
    
    print("Predictions completed.")

    print("\n--- 2. Generating Classification Report Image & Text ---")
    plot_and_save_classification_report(y_test_encoded, y_pred_encoded, label_encoder)

    print("\n--- 3. Generating Confusion Matrix Plot ---")
    plot_and_save_confusion_matrix(y_test_encoded, y_pred_encoded, label_encoder)
    
    print("\n--- Data Interpretation Script Finished Successfully ---")

if __name__ == "__main__":
    main()