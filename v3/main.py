import pandas as pd
from sklearn.preprocessing import LabelEncoder
import Model.preprocessing as pre
import Model.feature_engineering as fe
import Model.model_training as md
import joblib
import os
import kagglehub
import shutil
from typing import Optional, List, Tuple # Added Tuple and List for good practice

def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)

def get_target_variable(df: pd.DataFrame, target_column_name: str) -> Tuple[pd.Series, LabelEncoder]: # Changed to Tuple
    """Encodes the target variable using LabelEncoder and saves the encoder."""
    le = LabelEncoder()
    y = le.fit_transform(df[target_column_name])
    print(f"\nTarget variable '{target_column_name}' label encoded.")
    print(f"Classes: {le.classes_}")
    os.makedirs('pkl', exist_ok=True)
    joblib.dump(le, 'pkl/label_encoder.pkl')
    return y, le

def EDA(df: pd.DataFrame, numeric_cols_for_eda: List[str]): # Changed to List[str]
    """Performs and prints basic Exploratory Data Analysis on the DataFrame."""
    valid_numeric_cols = [col for col in numeric_cols_for_eda if col in df.columns]

    print("\n--- Exploratory Data Analysis (Post-processing) ---")
    print("Shape of final features (X):", df.shape)
    # print("Columns in final features (X):", df.columns.tolist()) # Can be too verbose
    
    print("\nFirst 5 rows of final features (X):")
    print(df.head())
    
    if valid_numeric_cols:
        print("\nSummary statistics for numeric columns in final features (X):")
        print(df[valid_numeric_cols].describe())
    else:
        print("\nNo numeric columns to describe in final features (X).")

    print("\nTotal missing values remaining in final features (X):", df.isnull().sum().sum())
    print("--- End of EDA ---")

def fetch_dataset(main_dir: str, data_subdir: str, filename: str) -> Optional[pd.DataFrame]: # FIXED HERE
    """
    Fetches the dataset, downloading from Kaggle if not found locally.
    Ensure Kaggle API credentials (kaggle.json) are set up in your environment
    (e.g., in ~/.kaggle/kaggle.json or by setting KAGGLE_USERNAME and KAGGLE_KEY env vars).
    """
    filepath = os.path.join(main_dir, data_subdir, filename)
    data_dir = os.path.join(main_dir, data_subdir)
    
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(filepath):
        print(f"Dataset '{filename}' found at '{filepath}'")
        data = load_data(filepath)
    else:
        print(f"File '{filepath}' not found, attempting to download 'mexwell/steamgames' from Kaggle...")
        try:
            download_path = kagglehub.dataset_download('mexwell/steamgames')
            
            source_file_to_copy = None
            possible_filenames = ['steam.csv', 'games.csv'] 
            
            for root, _, files in os.walk(download_path):
                for file_in_archive in files:
                    if file_in_archive.lower() in possible_filenames:
                        source_file_to_copy = os.path.join(root, file_in_archive)
                        print(f"Found CSV in Kaggle download: {source_file_to_copy}")
                        break
                if source_file_to_copy:
                    break
            
            if source_file_to_copy and os.path.exists(source_file_to_copy):
                shutil.copyfile(source_file_to_copy, filepath)
                print(f"Dataset copied to '{filepath}'")
                data = load_data(filepath)
            else:
                print(f"Could not find a suitable CSV file (e.g., 'steam.csv') in downloaded path: {download_path}")
                raise FileNotFoundError("CSV not found after Kaggle download attempt.")
        except Exception as e:
            print(f"Error during Kaggle download or file handling: {e}")
            print(f"Please ensure '{filename}' is manually placed in '{data_dir}' directory or check Kaggle setup.")
            return None
            
    return data

def main():
    # --- Configuration ---
    project_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_subdir = 'Data'
    raw_data_filename = 'steam.csv'
    processed_data_dir = 'Processed_Data'
    model_artefacts_dir = 'pkl'
    
    target_variable_name = 'Estimated owners'
    
    anova_pval_threshold = 0.05
    chi_square_pval_threshold = 0.05

    run_hyperparameter_tuning = True 
    tuning_iterations = 10 
    tuning_cv_folds = 2    

    # --- Setup Directories ---
    os.makedirs(os.path.join(project_dir, data_subdir), exist_ok=True)
    os.makedirs(os.path.join(project_dir, processed_data_dir), exist_ok=True)
    os.makedirs(os.path.join(project_dir, model_artefacts_dir), exist_ok=True)

    # --- Load Data ---
    df_original = fetch_dataset(project_dir, data_subdir, raw_data_filename)
    if df_original is None:
        print("Exiting due to data loading failure.")
        return

    print("Initial data shape:", df_original.shape)

    # --- Target Variable ---
    if target_variable_name not in df_original.columns:
        print(f"Target variable '{target_variable_name}' not found in the dataset.")
        return
    y, label_encoder = get_target_variable(df_original, target_variable_name)
    
    df_features = df_original.drop(columns=[target_variable_name])

    # --- Preprocessing ---
    df_features = pre.drop_unnecessary_columns(df_features)
    print("Shape after dropping unnecessary columns:", df_features.shape)

    pre.find_null_values(df_features)
    df_features = pre.drop_high_missing_columns(df_features, threshold=50)

    numeric_cols, categorical_cols = pre.separate_column_types(df_features)

    df_features = pre.preprocess_dates(df_features)
    df_features = pre.impute_missing_values(df_features, numeric_cols, categorical_cols)
    df_features = pre.convert_platform_booleans(df_features)
    
    df_features = pre.preprocess_multilabel_columns(df_features)
    df_features, numeric_cols = pre.simplify_multihot_columns(df_features, numeric_cols)
    df_features, numeric_cols = pre.separate_dates(df_features, numeric_cols)
    df_features, numeric_cols = pre.create_game_age_feature(df_features, numeric_cols)
    
    print("Shape before normalization:", df_features.shape)
    print(f"Numeric columns before normalization: {numeric_cols}")

    X, scaler = pre.normalize_data(df_features, None, numeric_cols) # Pass None as target_variable_name
    print("Shape after normalization (X):", X.shape)

    # --- Feature Engineering ---
    current_numeric_in_X = [col for col in numeric_cols if col in X.columns]
    X, final_numeric_cols = fe.anova_test_numeric(X, y, current_numeric_in_X, p_value_threshold=anova_pval_threshold)
    print("Shape after ANOVA:", X.shape)
    print(f"Numeric columns after ANOVA: {final_numeric_cols}")

    X = fe.chi_square_test(X, y, final_numeric_cols, p_value_threshold=chi_square_pval_threshold)
    print("Shape after Chi-Square:", X.shape)

    # --- Save Preprocessing Artifacts ---
    if scaler is not None: # Ensure scaler was created
        joblib.dump(scaler, os.path.join(project_dir, model_artefacts_dir, 'scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(project_dir, model_artefacts_dir, 'feature_columns.pkl'))
    joblib.dump(final_numeric_cols, os.path.join(project_dir, model_artefacts_dir, 'numeric_columns_final.pkl'))

    # --- Save Processed Data ---
    processed_data_for_csv = X.copy()
    processed_data_for_csv[target_variable_name] = y 
    processed_csv_path = os.path.join(project_dir, processed_data_dir, 'processed_steam.csv')
    processed_data_for_csv.to_csv(processed_csv_path, index=False)
    print(f"Processed data saved to '{processed_csv_path}'")
    
    # --- Data Splitting ---
    X_train, X_test, y_train, y_test = pre.split_data(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Preprocessing and Feature Engineering Complete! ---")

    # --- Final Check (EDA on processed X_train) ---
    numeric_cols_for_eda = [col for col in final_numeric_cols if col in X_train.columns]
    EDA(X_train, numeric_cols_for_eda) 
    
    # --- Model Training ---
    best_rf_params = {
        'n_estimators': 200, 
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    }

    if run_hyperparameter_tuning:
        print("\n--- Hyperparameter Tuning ---")
        tuned_params = md.tune_random_forest_hyperparameters(
            X_train, y_train, 
            n_iter=tuning_iterations, 
            cv=tuning_cv_folds, 
            random_state=42
        )
        best_rf_params.update(tuned_params)

    print("\n--- Model Training ---")
    model = md.train_model(X_train, y_train, model_type='random_forest', params=best_rf_params, random_state=42)

    # --- Feature Importances ---
    md.print_feature_importances(model, X_train.columns.tolist(), top_n=20)

    # --- Model Evaluation ---
    print("\n--- Model Evaluation on Test Set ---")
    mse_test, r2_test = md.evaluate_model(model, X_test, y_test)
    
    print("\n--- Cross-Validation on Training Set ---")
    mean_cv_mse, mean_cv_r2 = md.cross_validate_model(model, X_train, y_train, cv=3)
    
    # --- Save Model ---
    model_save_path = os.path.join(project_dir, model_artefacts_dir, 'ForecaSteam.pkl')
    md.save_model(model, model_save_path)

if __name__ == "__main__":
    main()