import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

# Load data
def load_data(filepath):
    return pd.read_csv(filepath)

def get_target_variable(df, target_column='Estimated owners'):
    y = df[target_column]
    le = LabelEncoder()

    return le.fit_transform(y) 

def drop_unnecessary_columns(df):
    # Drop the columns we’ll no longer need
    cols_to_drop = [
        'AppID', 'Name', 'About the game', 'Header image',
        'Website', 'Support url', 'Support email', 'Notes',
        'Screenshots', 'Movies', 'Metacritic url', 'Reviews',
        'Publishers', 'Developers'
    ]
    
    return df.drop(columns=cols_to_drop)

def find_null_values(df):
    missing_counts = df.isnull().sum()
    missing_frac = (missing_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_pct': missing_frac.round(2)
    }).sort_values('missing_pct', ascending=False)
    
    print("\nMissing value summary:")
    print(missing_summary)

def drop_high_missing_columns(df, threshold=50):
    # Drop columns with too much missing data (>50%)
    missing_counts = df.isnull().sum()
    missing_frac = (missing_counts / len(df)) * 100
    high_missing = missing_frac[missing_frac > threshold].index.tolist()
    
    if high_missing:
        df = df.drop(columns=high_missing)
        print(f"\nDropped columns >{threshold}% missing: {high_missing}")
    
    return df

def separate_column_types(df):
    # Separate features by type
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    return numeric_cols, categorical_cols

def preprocess_dates(df):
    # Convert Release date to datetime
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    
    return df

def impute_missing_values(df, numeric_cols, categorical_cols):
    # Impute missing numeric and categorical values.
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

def convert_platform_booleans(df):
    # Convert platform boolean columns to integers.
    for col in ['Windows', 'Mac', 'Linux']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df

def handle_multilabel_column(df, col_name, prefix):
    # Handle missing or invalid 'Categories' data
    df[col_name] = df[col_name].apply(lambda x: [] if pd.isna(x) else x.split(',') if isinstance(x, str) else [])

    # Multi-hot encode Categories using sparse matrix
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df[col_name])
    sparse = csr_matrix(encoded)    # Saves memory by compressing matrix, storing only non-zero entries

    encoded_df = pd.DataFrame.sparse.from_spmatrix(
        sparse,
        columns=[f"{prefix}_{item.strip()}" for item in mlb.classes_],
        index=df.index
    )

    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=[col_name], inplace=True)

    return df

def preprocess_multilabel_columns(df):
    # Apply multi-hot encoding to all relevant multilabel columns.
    multilabel_columns = {
        'Categories': 'Category',
        'Tags': 'Tag',
        'Genres': 'Genre',
        'Full audio languages': 'Audio',
        'Supported languages': 'Lang'
    }
    
    for col, prefix in multilabel_columns.items():
        if col in df.columns:
            df = handle_multilabel_column(df, col, prefix)
    return df


def main():
    # Set the target variable
    filepath = 'steam.csv'
    df = load_data(filepath)

    y = get_target_variable(df)
    df = drop_unnecessary_columns(df)
    
    # Peek at the cleaned frame
    print("Shape after dropping columns:", df.shape)
    print(df.head())

    find_null_values(df)

    df = drop_high_missing_columns(df, threshold=50)

    numeric_cols, categorical_cols = separate_column_types(df)

    df = preprocess_dates(df)
    df = impute_missing_values(df, numeric_cols, categorical_cols)
    df = convert_platform_booleans(df)
    df = preprocess_multilabel_columns(df)

    print("\nPreprocessing complete!")
    print("Final shape:", df.shape)
    print("Columns now:", df.columns.tolist())
    print(df.head())
    print("\nTotal missing values remaining:", df.isnull().sum().sum())

    
if __name__ == "__main__":
    main()