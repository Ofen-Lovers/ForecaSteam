import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

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
        
        # Add probability columns for each class
        for i, class_name in enumerate(label_encoder.classes_):
            results[f'probability_{class_name}'] = probabilities[:, i]
        
        # Save results
        print(f"Saving prediction results to {results_path}...")
        results.to_csv(results_path, index=False)
        print("Prediction completed successfully!")
        
        return results
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Use command line argument for test file if provided
    test_file = 'test1.csv'
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    perform_prediction(test_file) 