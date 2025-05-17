import os
import sys
import pandas as pd
import joblib
import random
import numpy as np
from datetime import datetime

def generate_test_data(num_samples=5, output_file='test1.csv'):
    """
    Generate test data for prediction. This creates sample data that 
    matches the format expected by the model.
    
    Args:
        num_samples (int): Number of test samples to generate
        output_file (str): Output file name
    
    Returns:
        pd.DataFrame: The generated test data
    """
    print(f"Generating {num_samples} test samples...")
    
    # Get directory paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prediction_dir = os.path.join(project_dir, 'Prediction')
    processed_data_path = os.path.join(project_dir, 'Processed_Data', 'processed_steam.csv')
    
    # Create output path
    output_path = os.path.join(prediction_dir, output_file)
    
    try:
        # Load feature columns from model artifacts
        pkl_dir = os.path.join(project_dir, 'pkl')
        feature_columns = joblib.load(os.path.join(pkl_dir, 'feature_columns.pkl'))
        
        # Try to load a sample of processed data first
        try:
            # Load just a few rows from the processed data
            df_sample = pd.read_csv(processed_data_path, nrows=10)
            if 'Estimated owners' in df_sample.columns:
                df_sample = df_sample.drop(columns=['Estimated owners'])
            
            # Use the sample data to create test data
            test_data = df_sample.sample(n=min(num_samples, len(df_sample)), replace=True).reset_index(drop=True)
            
            # For numeric columns, add some random variation
            numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                test_data[col] = test_data[col].apply(lambda x: max(0, x + random.uniform(-0.2*abs(x), 0.2*abs(x)) if x != 0 else x))
            
        except Exception as e:
            print(f"Could not load sample data: {e}")
            print("Creating synthetic test data from scratch...")
            
            # Create empty DataFrame with required columns
            test_data = pd.DataFrame(columns=feature_columns)
            
            # Fill with random data - this is very simplistic
            for col in feature_columns:
                if col.startswith(('Category_', 'Tag_', 'Genre_')):
                    # Binary features - mostly 0s with some 1s
                    test_data[col] = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
                else:
                    # Numeric features - random values between 0 and 100
                    test_data[col] = np.random.uniform(0, 100, size=num_samples)
            
        # Save test data
        test_data.to_csv(output_path, index=False)
        print(f"Test data saved to {output_path}")
        return test_data
        
    except Exception as e:
        print(f"Error generating test data: {e}")
        import traceback
        traceback.print_exc()
        return None

def perform_prediction(test_file='test1.csv'):
    """
    Perform predictions using the ForecaSteam_Classifier model.
    
    Args:
        test_file (str): Name of the test CSV file in the Prediction directory.
    
    Returns:
        pd.DataFrame: DataFrame with original data and prediction results.
    """
    # Get directory paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prediction_dir = os.path.join(project_dir, 'Prediction')
    pkl_dir = os.path.join(project_dir, 'pkl')
    
    # Create paths
    test_path = os.path.join(prediction_dir, test_file)
    results_path = os.path.join(prediction_dir, test_file.replace('.csv', '_predicted_results.csv'))
    
    # Check if test file exists
    if not os.path.exists(test_path):
        print(f"Error: Test file '{test_path}' not found!")
        return None
    
    try:
        # Load model artifacts
        print("Loading model artifacts...")
        model = joblib.load(os.path.join(pkl_dir, 'ForecaSteam_Classifier.pkl'))
        feature_columns = joblib.load(os.path.join(pkl_dir, 'feature_columns.pkl'))
        scaler = joblib.load(os.path.join(pkl_dir, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(pkl_dir, 'label_encoder.pkl'))
        
        # Load test data
        print(f"Loading test data from {test_path}...")
        test_data = pd.read_csv(test_path)
        print(f"Test data shape: {test_data.shape}")
        
        # Check for feature columns
        missing_columns = [col for col in feature_columns if col not in test_data.columns]
        if missing_columns:
            print(f"Warning: {len(missing_columns)} columns missing from test data. Adding with zero values.")
            for col in missing_columns:
                test_data[col] = 0
                
        # Ensure all required features are present in the correct order
        X_test = test_data[feature_columns].copy()
        print(f"Prepared test features shape: {X_test.shape}")
        
        # Make predictions
        print("Making predictions...")
        y_pred_numeric = model.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred_numeric)
        
        # Get prediction probabilities for all classes
        probabilities = model.predict_proba(X_test)
        
        # Create results DataFrame
        results = test_data.copy()
        results['predicted_class'] = y_pred_numeric
        results['predicted_owner_range'] = y_pred_labels
        
        # Add top 3 probabilities for each prediction
        class_names = label_encoder.classes_
        for i in range(len(results)):
            # Get top 3 class probabilities
            top_indices = np.argsort(probabilities[i])[::-1][:3]
            
            # Add columns for top 3 classes and their probabilities
            for j, idx in enumerate(top_indices):
                results.at[i, f'top{j+1}_class'] = class_names[idx]
                results.at[i, f'top{j+1}_probability'] = probabilities[i, idx]
        
        # Save results
        print(f"Saving prediction results to {results_path}...")
        results.to_csv(results_path, index=False)
        print("Prediction completed successfully!")
        
        # Print prediction summary
        print("\nPrediction Summary:")
        for i, (label, prob) in enumerate(zip(y_pred_labels, probabilities)):
            top_class_idx = np.argmax(prob)
            print(f"Sample {i+1}: Predicted '{label}' with {prob[top_class_idx]:.2f} probability")
        
        return results
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the entire prediction process."""
    test_file = 'test1.csv'
    
    # Generate test data
    test_data = generate_test_data(num_samples=5, output_file=test_file)
    if test_data is None:
        print("Failed to generate test data. Exiting.")
        return
    
    # Perform prediction
    results = perform_prediction(test_file)
    if results is None:
        print("Failed to perform prediction. Exiting.")
        return
    
    print(f"\nPrediction process completed successfully!")

if __name__ == "__main__":
    main() 