import joblib
import os
import pandas as pd

def load_model_artifacts():
    """Load the saved model and related artifacts."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_dir = os.path.join(project_dir, 'pkl')
    
    # Load model and related artifacts
    model = joblib.load(os.path.join(pkl_dir, 'ForecaSteam_Classifier.pkl'))
    feature_columns = joblib.load(os.path.join(pkl_dir, 'feature_columns.pkl'))
    numeric_columns = joblib.load(os.path.join(pkl_dir, 'numeric_columns_final.pkl'))
    scaler = joblib.load(os.path.join(pkl_dir, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(pkl_dir, 'label_encoder.pkl'))
    
    return model, feature_columns, numeric_columns, scaler, label_encoder

if __name__ == "__main__":
    model, feature_columns, numeric_columns, scaler, label_encoder = load_model_artifacts()
    
    print("Model type:", type(model).__name__)
    print(f"\nFeature columns ({len(feature_columns)}):")
    for col in feature_columns:
        print(f"- {col}")
    
    print(f"\nNumeric columns ({len(numeric_columns)}):")
    for col in numeric_columns:
        print(f"- {col}")
    
    print("\nClass labels (Estimated owners):")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"- Class {idx}: {label}") 