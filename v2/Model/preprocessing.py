import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from collections import Counter # Import Counter

def drop_unnecessary_columns(df):
    # Drop the columns we’ll no longer need
    cols_to_drop = [
        'AppID', 'Name', 'About the game', 'Header image',
        'Website', 'Support url', 'Support email', 'Notes',
        'Screenshots', 'Movies', 'Metacritic url', 'Reviews',
        'Publishers', 'Developers'
    ]
    # Drop columns if they exist, avoid KeyError
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df

def find_null_values(df):
    missing_counts = df.isnull().sum()
    missing_frac = (missing_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_pct': missing_frac.round(2)
    }).sort_values('missing_pct', ascending=False)
    
    print("\nMissing value summary:")
    print(missing_summary[missing_summary['missing_count'] > 0]) # Only show columns with missing values

def drop_high_missing_columns(df, threshold=50):
    # Drop columns with too much missing data (>threshold%)
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

    # dense_numeric_cols are not explicitly used later, but good for understanding
    dense_numeric_cols = [col for col in numeric_cols if not pd.api.types.is_sparse(df[col])]

    print(f"\nInitial numeric columns: {numeric_cols}")
    print(f"Initial categorical columns: {categorical_cols}")
    
    return numeric_cols, categorical_cols

def preprocess_dates(df):
    # Convert Release date to datetime
    if 'Release date' in df.columns:
        df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    return df

def impute_missing_values(df, numeric_cols, categorical_cols):
    # Impute missing numeric and categorical values.
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Impute only if columns exist and have missing values
    if numeric_cols:
        numeric_cols_in_df = [col for col in numeric_cols if col in df.columns]
        if numeric_cols_in_df:
            df[numeric_cols_in_df] = num_imputer.fit_transform(df[numeric_cols_in_df])

    if categorical_cols:
        categorical_cols_in_df = [col for col in categorical_cols if col in df.columns]
        if categorical_cols_in_df:
            df[categorical_cols_in_df] = cat_imputer.fit_transform(df[categorical_cols_in_df])
    
    return df

def convert_platform_booleans(df):
    # Convert platform boolean columns to integers.
    for col in ['Windows', 'Mac', 'Linux']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

def handle_multilabel_column(df, col_name, prefix):
    # Handle missing or invalid 'Categories' data
    df[col_name] = df[col_name].apply(lambda x: [] if pd.isna(x) else str(x).split(',') if isinstance(x, (str, int, float)) else [])

    # Multi-hot encode Categories using sparse matrix
    mlb = MultiLabelBinarizer(sparse_output=True) # Use sparse_output=True for direct sparse matrix
    encoded = mlb.fit_transform(df[col_name])
    # sparse = csr_matrix(encoded) # Not needed if sparse_output=True

    encoded_df = pd.DataFrame.sparse.from_spmatrix(
        encoded, # Use encoded directly
        columns=[f"{prefix}_{item.strip().replace(' ', '_').replace('&', 'And')}" for item in mlb.classes_], # Sanitize column names
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
    audio_cols = [col for col in df.columns if col.startswith('Audio_')]
    lang_cols = [col for col in df.columns if col.startswith('Lang_')]

    # Create new features for counts
    if audio_cols:
        df['Num_Audio_Languages'] = df[audio_cols].sum(axis=1)
    else:
        df['Num_Audio_Languages'] = 0 # Handle case where no audio_cols were created (e.g. column dropped)
        
    if lang_cols:
        df['Num_Supported_Languages'] = df[lang_cols].sum(axis=1)
    else:
        df['Num_Supported_Languages'] = 0

    # Drop the original detailed columns
    cols_to_drop_simplified = []
    if audio_cols:
        cols_to_drop_simplified.extend(audio_cols)
    if lang_cols:
        cols_to_drop_simplified.extend(lang_cols)
    
    df.drop(columns=cols_to_drop_simplified, inplace=True, errors='ignore')

    # Update numeric_cols list
    updated_numeric_cols = list(numeric_cols) # Create a copy
    if 'Num_Audio_Languages' not in updated_numeric_cols:
        updated_numeric_cols.append('Num_Audio_Languages')
    if 'Num_Supported_Languages' not in updated_numeric_cols:
        updated_numeric_cols.append('Num_Supported_Languages')
    
    # Remove original numeric columns if they were part of audio/lang (should not happen with current setup)
    updated_numeric_cols = [col for col in updated_numeric_cols if col not in cols_to_drop_simplified]

    print(f"Updated numeric features after simplification: {updated_numeric_cols}")
    return df, updated_numeric_cols

def separate_dates(df, numeric_cols):
    # Extract year, month, day from 'Release date'
    if 'Release date' in df.columns:
        df['Release_date_year'] = df['Release date'].dt.year
        df['Release_date_month'] = df['Release date'].dt.month
        df['Release_date_day'] = df['Release date'].dt.day

        # Add new date features to numeric_cols if not already present
        for col in ['Release_date_year', 'Release_date_month', 'Release_date_day']:
            if col not in numeric_cols:
                numeric_cols.append(col)
        
        # Drop the original 'Release date' column
        df.drop(columns=['Release date'], inplace=True)
        print(f"Date features created. Numeric_cols now: {numeric_cols}")
    return df, numeric_cols

def create_game_age_feature(df, numeric_cols):
    if 'Release_date_year' in df.columns:
        current_year = datetime.now().year
        df['Game_Age'] = current_year - df['Release_date_year']
        if 'Game_Age' not in numeric_cols:
            numeric_cols.append('Game_Age')
        print(f"Game_Age feature created. Numeric_cols now: {numeric_cols}")
    return df, numeric_cols

def normalize_data(df, target_variable_name, numeric_cols):
    # Ensure target_variable_name is not in features X
    features_df = df.copy()
    if target_variable_name in features_df.columns:
        features_df = features_df.drop(columns=[target_variable_name])
    
    # Ensure numeric_cols exist in df before scaling
    valid_numeric_cols = [col for col in numeric_cols if col in features_df.columns]
    
    scaler = StandardScaler()
    if valid_numeric_cols:
        features_df[valid_numeric_cols] = scaler.fit_transform(features_df[valid_numeric_cols])
        print(f"\nNormalized numeric columns: {valid_numeric_cols}")
    else:
        print("\nNo numeric columns found to normalize.")
        
    return features_df, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    # Check for classes with too few members for stratification
    y_counts = Counter(y)
    min_class_count = min(y_counts.values())
    
    stratify_param = y # Default to stratify
    
    # For train_test_split, the minimum members per class is effectively 2 (one for train, one for test at minimum)
    # More generally, for k-fold CV, it would be k.
    if min_class_count < 2:
        print(f"\nWARNING: The least populated class in y has only {min_class_count} member(s). "
              "This is too few for stratified splitting with the current test_size. "
              "Falling back to non-stratified splitting for this train/test split.")
        stratify_param = None
    else:
        print(f"\nProceeding with stratified splitting. Minimum class count in y: {min_class_count}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    print(f"\nData split complete: {len(X_train)} train samples, {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test