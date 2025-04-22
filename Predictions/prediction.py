import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix

class SteamOwnershipPredictor:
    def __init__(self):
        """Initialize by loading all required artifacts"""
        try:
            # Load all artifacts
            self.model = joblib.load('pkl/ForecaSteam.pkl')
            self.scaler = joblib.load('pkl/scaler.pkl')
            self.feature_columns = joblib.load('pkl/feature_columns.pkl')
            self.numeric_columns = joblib.load('pkl/numeric_columns.pkl')
            
            # Get the actual features used by the model
            self.model_features = (self.model.feature_names_in_ 
                                 if hasattr(self.model, 'feature_names_in_') 
                                 else self.feature_columns)
            
            # Filter numeric columns to only include those actually used by model
            self.model_numeric_cols = [col for col in self.numeric_columns 
                                     if col in self.model_features]
            
            # Get the features the scaler was actually trained on
            self.scaler_features = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else self.numeric_columns
            
            # Owner categories mapping
            self.owner_categories = [
                '0-20000', '20000-50000', '50000-100000', 
                '100000-200000', '200000-500000', '500000-1000000',
                '1000000-2000000', '2000000-5000000', '5000000-10000000',
                '10000000-20000000', '20000000-50000000', '50000000-100000000',
                '100000000-200000000'
            ]
            
            print("Prediction system loaded successfully!")
        except Exception as e:
            print(f"Error loading prediction system: {str(e)}")
            raise

    def _create_empty_feature_df(self):
        """Create an empty DataFrame with all expected features"""
        # Initialize with all model features
        empty_df = pd.DataFrame(columns=self.model_features)
        
        # Set default values
        for col in self.model_features:
            if col in self.model_numeric_cols:
                empty_df[col] = 0.0  # Default for numeric
            else:
                empty_df[col] = False  # Default for one-hot encoded
        
        return empty_df

    def preprocess_input(self, input_data):
        """Preprocess new data to match training format"""
        # Convert to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Create a template with all expected features
        processed_data = self._create_empty_feature_df()
        
        # Fill in available data from input
        for col in input_data.columns:
            if col in processed_data.columns:
                processed_data[col] = input_data[col]
        
        # Ensure correct data types for numeric columns
        for col in self.model_numeric_cols:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
        
        # Create a DataFrame with the exact features the scaler expects
        scaling_data = pd.DataFrame(columns=self.scaler_features)
        
        # Fill in available numeric data
        for col in self.scaler_features:
            if col in processed_data.columns:
                scaling_data[col] = processed_data[col]
            else:
                # Fill missing scaler features with 0 (like 'User score')
                scaling_data[col] = 0
        
        # Scale the numeric columns
        scaled_values = self.scaler.transform(scaling_data)
        
        # Put scaled values back into processed_data
        for i, col in enumerate(self.scaler_features):
            if col in processed_data.columns:
                processed_data[col] = scaled_values[:, i]
        
        return processed_data[self.model_features]

    def predict(self, input_data):
        """Make prediction on new data"""
        try:
            # Preprocess the input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            predicted_class = self.owner_categories[int(prediction)]
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(processed_data)[0]
                predicted_proba = {self.owner_categories[i]: float(prob) 
                                 for i, prob in enumerate(proba)}
            else:
                predicted_proba = None
            
            return {
                'predicted_class': predicted_class,
                'predicted_proba': predicted_proba,
                'all_predictions': prediction
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

def predict_from_csv(csv_path):
    """Helper function to make predictions from a CSV file"""
    predictor = SteamOwnershipPredictor()
    test_data = pd.read_csv(csv_path)
    
    # Remove target column if present
    if 'Estimated_owners' in test_data.columns:
        test_data = test_data.drop(columns=['Estimated_owners'])
    
    results = []
    for _, row in test_data.iterrows():
        result = predictor.predict(row.to_dict())
        results.append(result['predicted_class'])
    
    return results

if __name__ == "__main__":
    # Example usage with the test.csv file
    csv_path = "Processed_Data/test.csv"  # Update with your actual path
    predictions = predict_from_csv(csv_path)
    
    print("\nSteam Game Ownership Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"Game {i}: {pred}")