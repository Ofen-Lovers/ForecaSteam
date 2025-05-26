import pandas as pd
import os
import joblib
import numpy as np
import random

def generate_synthetic_unscaled_data(num_samples=5, feature_columns_list=None, numeric_columns_list=None):
    """
    Generates synthetic unscaled test data.
    
    Args:
        num_samples (int): Number of test samples to generate.
        feature_columns_list (list): List of all feature columns the model expects.
        numeric_columns_list (list): List of numeric columns (used to guide data generation).

    Returns:
        pd.DataFrame: The generated synthetic test data.
    """
    if feature_columns_list is None:
        print("Error: feature_columns_list is required to generate synthetic data.")
        return None

    print(f"Generating {num_samples} synthetic unscaled test samples...")
    test_data = pd.DataFrame(columns=feature_columns_list)
    
    for _ in range(num_samples):
        sample_row = {}
        for col in feature_columns_list:
            # Heuristics for generating data; might need refinement for better realism
            if col.startswith(('Category_', 'Tag_', 'Genre_')): # Binary multi-hot encoded features
                sample_row[col] = np.random.choice([0, 1], p=[0.85, 0.15]) # Most are 0
            elif col == 'Release date_year':
                sample_row[col] = np.random.randint(2000, 2025)
            elif col == 'Release date_month':
                sample_row[col] = np.random.randint(1, 13)
            elif col == 'Release date_day':
                sample_row[col] = np.random.randint(1, 29)
            elif col == 'Required age':
                sample_row[col] = np.random.choice([0, 6, 12, 18])
            elif col == 'Peak CCU':
                sample_row[col] = np.random.randint(0, 100000)
            elif col == 'Price':
                sample_row[col] = round(np.random.uniform(0, 70), 2)
            elif col == 'DLC count':
                sample_row[col] = np.random.randint(0, 10)
            elif col == 'Metacritic score':
                sample_row[col] = np.random.randint(0,101) if random.random() > 0.3 else 0 # Often 0 if missing
            elif col in ['Positive', 'Negative', 'Recommendations', 'Achievements']:
                sample_row[col] = np.random.randint(0, 50000)
            elif 'playtime' in col.lower(): # average/median playtime
                sample_row[col] = np.random.randint(0, 2000)
            elif 'Num_Audio_Languages' in col or 'Num_Supported_Languages' in col:
                sample_row[col] = np.random.randint(1, 20)
            elif col in ['Windows', 'Mac', 'Linux']: # Platform booleans (0 or 1)
                sample_row[col] = np.random.choice([0, 1])
            else: # Default for other numeric features or unrecognized ones
                # Check if it's a known numeric column, otherwise assume it might be a rare binary flag
                if numeric_columns_list and col in numeric_columns_list:
                    sample_row[col] = np.random.uniform(0, 100) # Generic numeric
                else: # Could be a less common binary/categorical one-hot encoded feature
                    sample_row[col] = np.random.choice([0,1], p=[0.95, 0.05])
        
        test_data_row = pd.DataFrame([sample_row])
        test_data = pd.concat([test_data, test_data_row], ignore_index=True)

    print(f"Generated synthetic test data with shape: {test_data.shape}")
    return test_data

def create_test_file(output_filename='test1.csv', num_samples=5):
    """
    Creates a test CSV file with synthetic unscaled data.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prediction_output_dir = os.path.join(project_dir, 'Predictions')
    os.makedirs(prediction_output_dir, exist_ok=True)
    
    pkl_dir = os.path.join(project_dir, 'pkl')
    
    try:
        feature_columns = joblib.load(os.path.join(pkl_dir, 'feature_columns.pkl'))
        # numeric_columns = joblib.load(os.path.join(pkl_dir, 'numeric_columns.pkl')) # For more targeted generation
    except FileNotFoundError:
        print(f"Error: 'feature_columns.pkl' not found in '{pkl_dir}'. Run main.py first.")
        return None
    
    test_df = generate_synthetic_unscaled_data(num_samples=num_samples, 
                                               feature_columns_list=feature_columns)
    
    if test_df is not None:
        output_path = os.path.join(prediction_output_dir, output_filename)
        test_df.to_csv(output_path, index=False)
        print(f"Synthetic unscaled test data ({num_samples} rows) saved to '{output_path}'")
        print("\nTest data columns:")
        for col in test_df.columns[:10]: # Print first 10
            print(f"- {col}")
        if len(test_df.columns) > 10:
            print("...")
        return output_path
    return None

if __name__ == "__main__":
    create_test_file(output_filename='sample_test_data.csv', num_samples=3)