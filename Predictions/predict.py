import pandas as pd
import joblib
import os
import numpy as np
import sys

from load_model_files import load_model_artifacts

def perform_prediction(test_file_path):
    """
    Perform predictions using the ForecaSteam (Regressor) model on data from test_file_path.
    The test data is expected to have unscaled numeric features.
    
    Args:
        test_file_path (str): Full path to the test CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with original data and prediction results, or None on error.
    """
    prediction_dir = os.path.dirname(test_file_path)
    input_filename_stem = os.path.splitext(os.path.basename(test_file_path))[0]
    results_filename = f"predicted_{input_filename_stem}.csv" # Generates predicted_test1.csv for test1.csv
    results_path = os.path.join(prediction_dir, results_filename)
    
    if not os.path.exists(test_file_path):
        print(f"Error: Test file '{test_file_path}' not found!")
        return None
    
    try:
        print("Loading model artifacts...")
        model, feature_columns, numeric_cols_for_scaling_from_pkl, scaler = load_model_artifacts()

        if not all([model, feature_columns, scaler]): # numeric_cols_for_scaling_from_pkl can be empty
            print("Failed to load one or more essential model artifacts (model, feature_columns, scaler). Cannot proceed.")
            return None
        if numeric_cols_for_scaling_from_pkl is None: # If pkl was missing or failed to load
            print("Warning: numeric_columns.pkl could not be loaded or was empty. Assuming no specific numeric columns list for scaler.")
            numeric_cols_for_scaling_from_pkl = []

        print(f"Loading test data from {test_file_path}...")
        test_data_raw = pd.read_csv(test_file_path)
        print(f"Raw test data shape: {test_data_raw.shape}")

        X_test_prepared = test_data_raw.copy()
        
        for col in feature_columns:
            if col not in X_test_prepared.columns:
                X_test_prepared[col] = 0 
        
        try:
            X_test_prepared = X_test_prepared[feature_columns]
        except KeyError as e:
            print(f"Error: Test data is missing one or more essential columns expected by the model: {e}")
            print(f"Model expects columns like: {feature_columns[:5]} ...")
            return None

        # --- Robust Scaling Block ---
        # numeric_cols_for_scaling_from_pkl is from numeric_columns.pkl. This list
        # defines which columns the scaler was *intended* to be fit on by main.py.
        
        # scaler.feature_names_in_ is what the scaler *actually* was fit on.
        # These two lists should ideally be identical.
        scaler_actually_expects = []
        if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
            scaler_actually_expects = list(scaler.feature_names_in_)
            print(f"Scaler object (scaler.pkl) reports it was fit on {len(scaler_actually_expects)} features like: {scaler_actually_expects[:5]}...")
        else:
            print("Warning: scaler.feature_names_in_ not available. Will rely solely on numeric_columns.pkl.")
            # If scaler.feature_names_in_ is not available, numeric_cols_for_scaling_from_pkl is our best guess.
            scaler_actually_expects = list(numeric_cols_for_scaling_from_pkl)


        if not scaler_actually_expects:
            print("No features identified as expected by the scaler. Skipping scaling.")
        else:
            print(f"Preparing data for scaler. Scaler expects {len(scaler_actually_expects)} features.")
            
            df_for_scaler_construction = {}
            for f_name in scaler_actually_expects:
                if f_name in X_test_prepared.columns:
                    df_for_scaler_construction[f_name] = X_test_prepared[f_name].copy()
                elif f_name in test_data_raw.columns:
                    print(f"Info: Feature '{f_name}' for scaling taken from raw test data (as it's expected by scaler but not in final model features).")
                    df_for_scaler_construction[f_name] = test_data_raw[f_name].copy()
                else:
                    print(f"Warning: Feature '{f_name}' (expected by scaler) not in prepared or raw test data. Using 0 for this feature during scaling.")
                    df_for_scaler_construction[f_name] = pd.Series(0, index=X_test_prepared.index, name=f_name)
            
            # Create DataFrame from dict, ensuring column order matches what scaler expects
            df_for_scaler = pd.DataFrame(df_for_scaler_construction)
            if not all(col in df_for_scaler.columns for col in scaler_actually_expects):
                missing_from_df = [col for col in scaler_actually_expects if col not in df_for_scaler.columns]
                print(f"Critical error: df_for_scaler is missing columns required by scaler: {missing_from_df}")
                return None

            df_for_scaler = df_for_scaler[scaler_actually_expects] # Ensure correct order

            print(f"Applying scaler to temporary DataFrame with {df_for_scaler.shape[1]} columns: {df_for_scaler.columns.tolist()[:5]}...")
            scaled_numeric_data = scaler.transform(df_for_scaler)
            scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=scaler_actually_expects, index=X_test_prepared.index)

            print(f"Updating final model input (X_test_prepared) with scaled values...")
            for f_name in scaler_actually_expects:
                if f_name in X_test_prepared.columns: # Only update if it's a final model feature
                    X_test_prepared[f_name] = scaled_numeric_df[f_name]
        # --- End of Robust Scaling Block ---

        print(f"Prepared and (potentially) scaled test features shape: {X_test_prepared.shape}")
        
        print("Making predictions...")
        y_pred_numeric = model.predict(X_test_prepared)
        
        results_df = test_data_raw.copy() 
        results_df['predicted_estimated_owners_value'] = y_pred_numeric
        
        print(f"Saving prediction results to {results_path}...")
        results_df.to_csv(results_path, index=False)
        print("Prediction completed successfully!")
        
        print("\nPrediction Summary (First 5 samples):")
        for i, pred_value in enumerate(y_pred_numeric[:5]):
            original_sample_info = []
            if 'Price' in test_data_raw.columns:
                 original_sample_info.append(f"Price: {test_data_raw.iloc[i].get('Price', 'N/A')}")
            if 'Peak CCU' in test_data_raw.columns:
                 original_sample_info.append(f"Peak CCU: {test_data_raw.iloc[i].get('Peak CCU', 'N/A')}")
            
            print(f"Sample {i+1} ({', '.join(original_sample_info)}): Predicted Value = {pred_value:,.0f}")
        if len(y_pred_numeric) > 5:
            print("...")

        return results_df
    
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        return None
    except ValueError as e:
        print(f"ValueError during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file_arg = sys.argv[1]
        if not os.path.isabs(test_file_arg):
             project_dir_from_predict = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             path_in_predictions = os.path.join(project_dir_from_predict, 'Predictions', test_file_arg)
             path_in_project_root = os.path.join(project_dir_from_predict, test_file_arg)

             if os.path.exists(path_in_predictions):
                 test_file_full_path = path_in_predictions
             elif os.path.exists(path_in_project_root):
                 test_file_full_path = path_in_project_root
             else:
                 test_file_full_path = test_file_arg 
        else:
            test_file_full_path = test_file_arg

        print(f"Attempting prediction on: {test_file_full_path}")
        perform_prediction(test_file_full_path)
    else:
        print("Usage: python predict.py <path_to_test_csv_file>")
        print("Example: python Predictions/predict.py Predictions/test1.csv")
        print("\nAlternatively, run 'run_prediction.py' to generate test data (e.g., test1.csv) and predict.")