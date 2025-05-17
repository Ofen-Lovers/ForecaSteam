import pandas as pd
import os
import random
import numpy as np
import joblib

def create_test_data(num_samples=5):
    """
    Create sample test data for prediction.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(project_dir, 'Processed_Data', 'processed_steam.csv')
    
    # Create output directory if it doesn't exist
    prediction_dir = os.path.join(project_dir, 'Prediction')
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Load a sample of processed data
    try:
        # Try to read processed data
        df_sample = pd.read_csv(processed_data_path, nrows=10)
        print(f"Loaded processed data sample with {len(df_sample)} rows and {len(df_sample.columns)} columns")
        
        # Remove the target column if it exists
        if 'Estimated_owners' in df_sample.columns:
            df_sample = df_sample.drop(columns=['Estimated_owners'])
            
        # Create test data by modifying some values
        test_data = df_sample.sample(n=min(num_samples, len(df_sample)), replace=True).reset_index(drop=True)
        
        # For numeric columns, slightly modify values
        numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            # Add small random variation to numeric columns
            test_data[col] = test_data[col].apply(lambda x: max(0, x + random.uniform(-0.1*abs(x), 0.1*abs(x)) if x != 0 else x))
        
        # Save test data to test1.csv
        test_path = os.path.join(prediction_dir, 'test1.csv')
        test_data.to_csv(test_path, index=False)
        print(f"Test data with {len(test_data)} rows saved to {test_path}")
        
        return test_data
        
    except Exception as e:
        print(f"Error loading processed data: {e}")
        print("Creating synthetic test data instead...")
        
        # Create synthetic data by loading feature columns
        try:
            pkl_dir = os.path.join(project_dir, 'pkl')
            feature_columns = joblib.load(os.path.join(pkl_dir, 'feature_columns.pkl'))
            
            # Create empty DataFrame with required columns
            test_data = pd.DataFrame(columns=feature_columns)
            
            # Fill with random data
            for col in feature_columns:
                if col.startswith(('Category_', 'Tag_', 'Genre_')):
                    # Binary features
                    test_data[col] = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
                else:
                    # Numeric features
                    test_data[col] = np.random.uniform(0, 10, size=num_samples)
            
            # Save test data
            test_path = os.path.join(prediction_dir, 'test1.csv')
            test_data.to_csv(test_path, index=False)
            print(f"Synthetic test data with {len(test_data)} rows saved to {test_path}")
            
            return test_data
            
        except Exception as e:
            print(f"Failed to create synthetic test data: {e}")
            return None

if __name__ == "__main__":
    create_test_data(num_samples=5)