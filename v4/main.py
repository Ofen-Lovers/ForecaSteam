import pandas as pd
from sklearn.preprocessing import LabelEncoder
import Model.preprocessing as pre
import Model.feature_engineering as fe
import Model.model_training as md
import joblib
import os
import kagglehub
import shutil
from typing import Optional, List, Tuple

# --- Configuration ---
IS_CLASSIFICATION_TASK = True # <<< SET TO TRUE FOR CLASSIFICATION
# ---

def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)

def get_target_variable(df: pd.DataFrame, target_column_name: str) -> Tuple[pd.Series, LabelEncoder]:
    """Encodes the target variable using LabelEncoder and saves the encoder."""
    le = LabelEncoder()
    y = le.fit_transform(df[target_column_name])
    print(f"\nTarget variable '{target_column_name}' label encoded.")
    print(f"Classes: {le.classes_}")
    # Ensure pkl directory exists
    os.makedirs('pkl', exist_ok=True)
    joblib.dump(le, 'pkl/label_encoder.pkl')
    return y, le

def EDA(df: pd.DataFrame, numeric_cols_for_eda: List[str]):
    """Performs and prints basic Exploratory Data Analysis on the DataFrame."""
    valid_numeric_cols = [col for col in numeric_cols_for_eda if col in df.columns]

    print("\n--- Exploratory Data Analysis (Post-processing) ---")
    print("Shape of final features (X):", df.shape)
    
    print("\nFirst 5 rows of final features (X):")
    print(df.head())
    
    if valid_numeric_cols:
        print("\nSummary statistics for numeric columns in final features (X):")
        print(df[valid_numeric_cols].describe())
    else:
        print("\nNo numeric columns to describe in final features (X).")

    print("\nTotal missing values remaining in final features (X):", df.isnull().sum().sum())
    print("--- End of EDA ---")

def fetch_dataset(main_dir: str, data_subdir: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Fetches the dataset, downloading from Kaggle if not found locally.
    Ensure Kaggle API credentials (kaggle.json) are set up in your environment.
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
            print(f"Please ensure '{filename}' is manually placed in '{data_dir}' or check Kaggle setup.")
            return None
            
    return data

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(project_dir), 'Data')
    processed_data_dir = os.path.join(project_dir, 'Processed_Data')
    model_artefacts_dir = os.path.join(project_dir, 'pkl')
    
    target_variable_name = 'Estimated owners'
    
    # Feature selection (ANOVA and Chi-Square) is generally for relevance to the target.
    # While ANOVA's f_classif is for classification/regression and Chi-Square for categorical features,
    # their use here for feature *reduction* prior to modeling is usually fine for both tasks.
    # The primary impact will be on the model's interpretation of the target.
    anova_pval_threshold = 0.05
    chi_square_pval_threshold = 0.05

    run_hyperparameter_tuning = True 
    tuning_iterations = 10 # Keep low for speed, increase for better tuning
    tuning_cv_folds = 2    # Keep low for speed

    os.makedirs(os.path.join(project_dir, data_dir), exist_ok=True)
    os.makedirs(os.path.join(project_dir, processed_data_dir), exist_ok=True)
    os.makedirs(os.path.join(project_dir, model_artefacts_dir), exist_ok=True)

    raw_data_path = os.path.join(data_dir, 'steam.csv')
    df_original = fetch_dataset(project_dir, data_dir, 'steam.csv')
    if df_original is None:
        print("Exiting due to data loading failure.")
        return

    print("Initial data shape:", df_original.shape)

    if target_variable_name not in df_original.columns:
        print(f"Target variable '{target_variable_name}' not found in the dataset.")
        return
    y, label_encoder_for_target = get_target_variable(df_original, target_variable_name) # Save LE
    
    df_features = df_original.drop(columns=[target_variable_name])

    # --- Preprocessing (largely same for classification vs regression features) ---
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
    X, scaler = pre.normalize_data(df_features, None, numeric_cols)
    print("Shape after normalization (X):", X.shape)

    # --- Feature Engineering (ANOVA and Chi-Square can still be used) ---
    # ANOVA's f_classif (used by SelectKBest, default for f_classif) works for classification targets.
    current_numeric_in_X = [col for col in numeric_cols if col in X.columns]
    X, final_numeric_cols = fe.anova_test_numeric(X, y, current_numeric_in_X, p_value_threshold=anova_pval_threshold)
    print("Shape after ANOVA:", X.shape)
    print(f"Numeric columns after ANOVA: {final_numeric_cols}")

    X = fe.chi_square_test(X, y, final_numeric_cols, p_value_threshold=chi_square_pval_threshold)
    print("Shape after Chi-Square:", X.shape)

    # --- Save Preprocessing Artifacts ---
    if scaler is not None:
        joblib.dump(scaler, os.path.join(model_artefacts_dir, 'scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(model_artefacts_dir, 'feature_columns.pkl'))
    joblib.dump(final_numeric_cols, os.path.join(model_artefacts_dir, 'numeric_columns_final.pkl'))

    processed_data_for_csv = X.copy()
    processed_data_for_csv[target_variable_name] = y 
    processed_csv_path = os.path.join(processed_data_dir, 'processed_steam.csv')
    processed_data_for_csv.to_csv(processed_csv_path, index=False)
    print(f"Processed data saved to '{processed_csv_path}'")
    
    X_train, X_test, y_train, y_test = pre.split_data(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Preprocessing and Feature Engineering Complete! ---")

    numeric_cols_for_eda = [col for col in final_numeric_cols if col in X_train.columns]
    EDA(X_train, numeric_cols_for_eda) 
    
    # --- Model Training (Classification) ---
    model_name_suffix = "Classifier" if IS_CLASSIFICATION_TASK else "Regressor"
    model_type_name = f"random_forest_{model_name_suffix.lower()}"

    default_params = { # General defaults, tuning will override
        'n_estimators': 200, 'max_depth': 30,
        'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'
    }
    if IS_CLASSIFICATION_TASK:
        default_params['class_weight'] = 'balanced' # Good starting point for imbalance

    best_model_params = default_params.copy()

    if run_hyperparameter_tuning:
        print("\n--- Hyperparameter Tuning ---")
        tuned_params = md.tune_random_forest_hyperparameters(
            X_train, y_train,
            is_classifier=IS_CLASSIFICATION_TASK, # Pass flag
            n_iter=tuning_iterations, 
            cv=tuning_cv_folds, 
            random_state=42
        )
        best_model_params.update(tuned_params)

    print("\n--- Model Training ---")
    model = md.train_model(X_train, y_train, 
                           model_type=model_type_name, 
                           params=best_model_params, 
                           random_state=42)

    md.print_feature_importances(model, X_train.columns.tolist(), top_n=20)

    # --- Model Evaluation ---
    print(f"\n--- Model Evaluation on Test Set ({model_name_suffix}) ---")
    # Pass label_encoder for class names in reports
    metrics1, metrics2 = md.evaluate_model(model, X_test, y_test, 
                                           is_classifier=IS_CLASSIFICATION_TASK,
                                           label_encoder=label_encoder_for_target) 
    
    print(f"\n--- Cross-Validation on Training Set ({model_name_suffix}) ---")
    cv_metric1, cv_metric2 = md.cross_validate_model(model, X_train, y_train, 
                                                     is_classifier=IS_CLASSIFICATION_TASK, cv=3)
    
    model_save_path = os.path.join(model_artefacts_dir, f'ForecaSteam_{model_name_suffix}.pkl')
    md.save_model(model, model_save_path)

if __name__ == "__main__":
    main()