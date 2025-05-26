import os
import sys
import pandas as pd
import joblib
import numpy as np
import random

# Import perform_prediction from the local predict.py
from predict import perform_prediction as local_perform_prediction
# Import generate_synthetic_unscaled_data from create_test_data.py
from create_test_data import generate_synthetic_unscaled_data

def setup_test_data(num_samples=5, output_filename='test1.csv'): # Changed default output_filename
    """
    Generates and saves synthetic unscaled test data for prediction.
    
    Args:
        num_samples (int): Number of test samples to generate.
        output_filename (str): Output CSV file name for the test data.
    
    Returns:
        str: Full path to the generated test data CSV file, or None on error.
    """
    print(f"Setting up test data: generating {num_samples} samples for '{output_filename}'...")
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prediction_output_dir = os.path.join(project_dir, 'Predictions')
    os.makedirs(prediction_output_dir, exist_ok=True)
    
    pkl_dir = os.path.join(project_dir, 'pkl')
    
    try:
        feature_columns_from_training = joblib.load(os.path.join(pkl_dir, 'feature_columns.pkl'))
        numeric_columns_scaled_in_training = joblib.load(os.path.join(pkl_dir, 'numeric_columns.pkl'))
        
        print("Creating synthetic unscaled test data...")
        test_df = generate_synthetic_unscaled_data(
            num_samples=num_samples,
            feature_columns_list=feature_columns_from_training,
            numeric_columns_list=numeric_columns_scaled_in_training
        )

        if test_df is None:
            print("Failed to generate synthetic test data.")
            return None
            
        output_path = os.path.join(prediction_output_dir, output_filename)
        test_df.to_csv(output_path, index=False)
        print(f"Synthetic unscaled test data saved to '{output_path}'")
        return output_path
        
    except FileNotFoundError as e:
        print(f"Error loading artifact for generating test data: {e}. File missing: {e.filename}")
        print(f"Ensure 'feature_columns.pkl' and 'numeric_columns.pkl' exist in '{pkl_dir}'. Run main.py.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while generating test data: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to generate test data and run the prediction process."""
    
    test_filename_to_generate = 'test1.csv' 
    num_test_samples = 5
    
    generated_test_file_path = setup_test_data(num_samples=num_test_samples, output_filename=test_filename_to_generate)
    
    if generated_test_file_path is None:
        print("Failed to generate test data. Exiting prediction process.")
        return
    
    print(f"\nTest data generated: '{generated_test_file_path}'")
    
    results_df = local_perform_prediction(test_file_path=generated_test_file_path)
    
    if results_df is None:
        print("\nPrediction process encountered an error.")
    else:
        # The output filename is now handled by perform_prediction to be predicted_{input_stem}.csv
        input_filename_stem = os.path.splitext(os.path.basename(generated_test_file_path))[0]
        expected_results_filename = f"predicted_{input_filename_stem}.csv"
        expected_results_path = os.path.join(os.path.dirname(generated_test_file_path), expected_results_filename)
        print(f"\nPrediction process completed. Results should be saved as '{expected_results_path}'")

if __name__ == "__main__":
    main()