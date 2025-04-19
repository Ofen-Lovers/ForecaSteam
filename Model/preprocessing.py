import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

    dense_numeric_cols = [col for col in numeric_cols if not pd.api.types.is_sparse(df[col])]

    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    return numeric_cols, categorical_cols, dense_numeric_cols

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

def simplify_multihot_columns(df, numeric_cols):
    # Identify multi-hot encoded columns
    audio_cols = [col for col in df.columns if col.startswith('Audio')]
    lang_cols = [col for col in df.columns if col.startswith('Lang')]

    # Create new features for counts
    df['Num_Audio_Languages'] = df[audio_cols].sum(axis=1)
    df['Num_Supported_Languages'] = df[lang_cols].sum(axis=1)

    # Drop the original detailed columns if you want
    df.drop(columns=audio_cols + lang_cols, inplace=True)

    new_numeric_cols = numeric_cols + ['Num_Audio_Languages', 'Num_Supported_Languages']

    # If you want to verify:
    print(f"Updated numeric features: {new_numeric_cols}")

    return df, new_numeric_cols

def seperate_dates(df):
    # Extract year, month, day from 'Release date'
    df['Release date_year'] = df['Release date'].dt.year
    df['Release date_month'] = df['Release date'].dt.month
    df['Release date_day'] = df['Release date'].dt.day

    # Drop the original 'Release date' column
    df.drop(columns=['Release date'], inplace=True)
    
    return df

def normalize_data(df, target_variable, numeric_cols):
    X = df.drop(columns=[target_variable])  # Drop target variable to separate features

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print("\nNormalized numeric columns.")

    return X, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    # 80% training set/20% test set, random seed of 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\nData split complete: {len(X_train)} train samples, {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test

