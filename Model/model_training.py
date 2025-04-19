from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(X_train, y_train, n_estimators=100, random_state=42, n_jobs=-1):
    model = RandomForestRegressor(n_estimators=n_estimators,random_state=random_state,n_jobs=n_jobs)
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")

    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return mse, r2

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename} successfully!")