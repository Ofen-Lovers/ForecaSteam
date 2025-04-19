import pandas as pd
from sklearn.preprocessing import LabelEncoder
import Model.preprocessing as pre
import Model.feature_engineering as fe
import Model.model_training as md

def load_data(filepath):
    return pd.read_csv(filepath)

def get_target_variable(target_variable):
    le = LabelEncoder()

    return le.fit_transform(target_variable) 

def EDA(df, numeric_cols):
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    print("Final shape:", df.shape)               # Quick overview of rows and columns
    print("Columns now:", df.columns.tolist())    # List of all column names
    print(df.dtypes)                              # Data types per column
    print(df.info())                              # More details: non-null counts, memory usage
    print(df.head())                              # First few rows to understand the data layout
    print(df[numeric_cols].describe())            # Summary stats for numeric columns

    print("\nTotal missing values remaining:", df.isnull().sum().sum())

def main():
    # Set the target variable
    filepath = 'Data/steam.csv'
    df = load_data(filepath)
    target_variable = 'Estimated owners'
    
    y = get_target_variable(df[target_variable])
    df = pre.drop_unnecessary_columns(df)
    print("Shape after dropping columns:", df.shape)    #Peak at columns after dropping
    print(df.head())

    pre.find_null_values(df)
    df = pre.drop_high_missing_columns(df, threshold=50)

    numeric_cols, categorical_cols, dense_numeric_cols = pre.separate_column_types(df)

    df = pre.preprocess_dates(df)
    df = pre.impute_missing_values(df, numeric_cols, categorical_cols)
    df = pre.convert_platform_booleans(df)
    df = pre.preprocess_multilabel_columns(df)
    df = pre.seperate_dates(df)
    X, scaler = pre.normalize_data(df, target_variable, numeric_cols)

    #Feature Engineering
    X = fe.anova_test_numeric(numeric_cols, X, y)
    X = fe.chi_square_test(df, target_variable, numeric_cols, X)
    print("\nShape after dropping columns:", df.shape)  

    X_train, X_test, y_train, y_test = pre.split_data(X, y, test_size=0.2, random_state=42)
    print("\nPreprocessing complete!")

    #Final Check
    EDA(X, numeric_cols)

    # Model Training
    model = md.train_model(X_train, y_train)
    mse, r2 = md.evaluate_model(model, X_test, y_test)
    # md.save_model(model, 'ForecaSteam.pkl')

if __name__ == "__main__":
    main()