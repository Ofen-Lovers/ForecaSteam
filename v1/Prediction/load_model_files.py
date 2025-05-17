import joblib
import os
import pandas as pd

def load_model_artifacts():
    """Load the saved model and related artifacts."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_dir = os.path.join(project_dir, 'pkl')
    
    # Load model and related artifacts
    model = joblib.load(os.path.join(pkl_dir, 'ForecaSteam.pkl'))
    feature_columns = joblib.load(os.path.join(pkl_dir, 'feature_columns.pkl'))
    numeric_columns = joblib.load(os.path.join(pkl_dir, 'numeric_columns.pkl'))
    scaler = joblib.load(os.path.join(pkl_dir, 'scaler.pkl'))
    
    return model, feature_columns, numeric_columns, scaler

if __name__ == "__main__":
    model, feature_columns, numeric_columns, scaler = load_model_artifacts()
    
    print("Model type:", type(model).__name__)
    print(f"\nFeature columns ({len(feature_columns)}):")
    for col in feature_columns[:10]:
        print(f"- {col}")
    if len(feature_columns) > 10:
        print(f"... and {len(feature_columns) - 10} more")
    
    print(f"\nNumeric columns ({len(numeric_columns)}):")
    for col in numeric_columns:
        print(f"- {col}")