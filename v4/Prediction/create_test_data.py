import pandas as pd
import os
import random

def create_test_data():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(project_dir, 'Processed_Data', 'processed_steam.csv')
    
    # Load a sample of processed data (first 5 rows)
    try:
        # Try to read only the needed columns
        df_sample = pd.read_csv(processed_data_path, nrows=5)
        print(f"Loaded processed data sample with {len(df_sample)} rows and {len(df_sample.columns)} columns")
        
        # Remove the target column if it exists
        if 'Estimated owners' in df_sample.columns:
            df_sample = df_sample.drop(columns=['Estimated owners'])
            
        # Create test data by modifying some values
        test_data = df_sample.copy()
        
        # For numeric columns, slightly modify values
        numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            # Add small random variation to numeric columns
            test_data[col] = test_data[col].apply(lambda x: max(0, x + random.uniform(-0.1*abs(x), 0.1*abs(x)) if x != 0 else x))
        
        # Save test data to test1.csv
        test_path = os.path.join(project_dir, 'Prediction', 'test1.csv')
        test_data.to_csv(test_path, index=False)
        print(f"Test data with {len(test_data)} rows saved to {test_path}")
        
        # Print column names
        print("\nTest data columns:")
        for col in test_data.columns:
            print(f"- {col}")
        
        return test_data
    
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    create_test_data() 