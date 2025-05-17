import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from collections import Counter
import ast 
import re
from typing import Optional, Tuple, List, Union # Added Optional, Tuple, List

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops predefined unnecessary columns from the DataFrame."""
    cols_to_drop = [
        'AppID', 'Name', 'About the game', 'Header image',
        'Website', 'Support url', 'Support email', 'Notes',
        'Screenshots', 'Movies', 'Metacritic url', 'Reviews',
        'Publishers', 'Developers'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    return df

def find_null_values(df: pd.DataFrame):
    """Prints a summary of missing values in the DataFrame."""
    missing_counts = df.isnull().sum()
    missing_frac = (missing_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_pct': missing_frac.round(2)
    }).sort_values('missing_pct', ascending=False)
    
    print("\nMissing value summary:")
    print(missing_summary[missing_summary['missing_count'] > 0])

def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
    """Drops columns with missing data percentage greater than the threshold."""
    missing_counts = df.isnull().sum()
    missing_frac = (missing_counts / len(df)) * 100
    high_missing = missing_frac[missing_frac > threshold].index.tolist()
    
    if high_missing:
        df = df.drop(columns=high_missing)
        print(f"\nDropped columns >{threshold}% missing: {high_missing}")
    
    return df

def separate_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]: # Changed to Tuple[List[str], List[str]]
    """Separates DataFrame columns into numeric and categorical types."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nInitial numeric columns: {numeric_cols}")
    print(f"Initial categorical columns: {categorical_cols}")
    
    return numeric_cols, categorical_cols

def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Converts 'Release date' column to datetime objects."""
    if 'Release date' in df.columns:
        df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    return df

def impute_missing_values(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame: # Changed to List[str]
    """Imputes missing values for numeric (median) and categorical (most frequent) columns."""
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    if numeric_cols:
        numeric_cols_in_df = [col for col in numeric_cols if col in df.columns]
        if numeric_cols_in_df:
            df[numeric_cols_in_df] = num_imputer.fit_transform(df[numeric_cols_in_df])

    if categorical_cols:
        categorical_cols_in_df = [col for col in categorical_cols if col in df.columns]
        if categorical_cols_in_df:
            df[categorical_cols_in_df] = cat_imputer.fit_transform(df[categorical_cols_in_df])
    
    return df

def convert_platform_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Converts platform boolean columns (Windows, Mac, Linux) to integers."""
    for col in ['Windows', 'Mac', 'Linux']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

def _sanitize_column_name(name: str) -> str:
    """Sanitizes a string to be a valid column name."""
    name = str(name).strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[&/\\:\-\(\)\[\]\{\}\'\"]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name

def smart_parse_multilabel(value_str: str, is_list_like_string: bool) -> List[str]: # Changed to List[str]
    """
    Parses a string value into a list of strings, handling both
    comma-separated values and stringified list representations.
    """
    if pd.isna(value_str):
        return []
    
    s_val = str(value_str).strip()
    if not s_val:
        return []

    if is_list_like_string:
        try:
            if s_val.startswith('[') and s_val.endswith(']'):
                s_val_inner = s_val[1:-1].strip()
                if not s_val_inner: 
                    return []
                try:
                    parsed_list = ast.literal_eval(s_val)
                    if isinstance(parsed_list, list):
                         return [_sanitize_column_name(str(item)) for item in parsed_list if str(item).strip()]
                    return [_sanitize_column_name(str(parsed_list))] if str(parsed_list).strip() else []
                except (ValueError, SyntaxError):
                    return [_sanitize_column_name(item) for item in s_val_inner.split(',') if item.strip()]
            else:
                 return [_sanitize_column_name(s_val)] if s_val else []
        except (ValueError, SyntaxError, TypeError):
            return [_sanitize_column_name(s_val)] if s_val else []
    else:
        return [_sanitize_column_name(item) for item in s_val.split(',') if item.strip()]


def handle_multilabel_column(df: pd.DataFrame, col_name: str, prefix: str, is_list_like_string_col: bool) -> pd.DataFrame:
    """
    Handles a multi-label column by parsing its string values,
    then applying multi-hot encoding.
    """
    df[col_name] = df[col_name].apply(lambda x: smart_parse_multilabel(x, is_list_like_string_col))

    mlb = MultiLabelBinarizer(sparse_output=True)
    encoded = mlb.fit_transform(df[col_name])
    
    sanitized_classes = [f"{prefix}_{_sanitize_column_name(item)}" for item in mlb.classes_]
    
    encoded_df = pd.DataFrame.sparse.from_spmatrix(
        encoded,
        columns=sanitized_classes,
        index=df.index
    )

    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=[col_name], inplace=True)
    return df

def preprocess_multilabel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies multi-hot encoding to all relevant multilabel columns."""
    multilabel_map = {
        'Categories': {'prefix': 'Category', 'is_list_like': False},
        'Tags': {'prefix': 'Tag', 'is_list_like': False},
        'Genres': {'prefix': 'Genre', 'is_list_like': False},
        'Full audio languages': {'prefix': 'Audio', 'is_list_like': True},
        'Supported languages': {'prefix': 'Lang', 'is_list_like': True}
    }
    
    for col, conf in multilabel_map.items():
        if col in df.columns:
            print(f"Processing multi-label column: {col}")
            df = handle_multilabel_column(df, col, conf['prefix'], conf['is_list_like'])
    return df

def simplify_multihot_columns(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]: # Changed
    """
    Simplifies multi-hot encoded language columns into counts
    (Num_Audio_Languages, Num_Supported_Languages) and updates numeric_cols.
    """
    audio_cols = [col for col in df.columns if col.startswith('Audio_')]
    lang_cols = [col for col in df.columns if col.startswith('Lang_')]

    if audio_cols:
        df['Num_Audio_Languages'] = df[audio_cols].sum(axis=1)
    else:
        df['Num_Audio_Languages'] = 0 
        
    if lang_cols:
        df['Num_Supported_Languages'] = df[lang_cols].sum(axis=1)
    else:
        df['Num_Supported_Languages'] = 0

    cols_to_drop_simplified = []
    if audio_cols:
        cols_to_drop_simplified.extend(audio_cols)
    if lang_cols:
        cols_to_drop_simplified.extend(lang_cols)
    
    df.drop(columns=cols_to_drop_simplified, inplace=True, errors='ignore')

    updated_numeric_cols = list(numeric_cols)
    if 'Num_Audio_Languages' not in updated_numeric_cols:
        updated_numeric_cols.append('Num_Audio_Languages')
    if 'Num_Supported_Languages' not in updated_numeric_cols:
        updated_numeric_cols.append('Num_Supported_Languages')
    
    updated_numeric_cols = [col for col in updated_numeric_cols if col not in cols_to_drop_simplified]

    print(f"Updated numeric features after simplification: {updated_numeric_cols}")
    return df, updated_numeric_cols

def separate_dates(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]: # Changed
    """Extracts year, month, day from 'Release date' and adds them to numeric_cols."""
    if 'Release date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Release date']):
        df['Release_date_year'] = df['Release date'].dt.year
        df['Release_date_month'] = df['Release date'].dt.month
        df['Release_date_day'] = df['Release date'].dt.day

        for col in ['Release_date_year', 'Release_date_month', 'Release_date_day']:
            if col not in numeric_cols:
                numeric_cols.append(col)
        
        df.drop(columns=['Release date'], inplace=True)
        print(f"Date features created. Numeric_cols now: {numeric_cols}")
    else:
        print("Skipping date separation: 'Release date' not found or not datetime type.")
    return df, numeric_cols

def create_game_age_feature(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]: # Changed
    """Creates 'Game_Age' feature from 'Release_date_year' and adds it to numeric_cols."""
    if 'Release_date_year' in df.columns:
        current_year = datetime.now().year
        df['Game_Age'] = current_year - df['Release_date_year']
        if 'Game_Age' not in numeric_cols:
            numeric_cols.append('Game_Age')
        print(f"Game_Age feature created. Numeric_cols now: {numeric_cols}")
    else:
        print("Skipping Game_Age creation: 'Release_date_year' not found.")
    return df, numeric_cols

# FIXED TYPE HINTS IN THIS FUNCTION SIGNATURE
def normalize_data(df: pd.DataFrame, target_variable_name: Optional[str], numeric_cols: List[str]) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Normalizes numeric columns using StandardScaler.
    Optionally drops target_variable_name if provided.
    """
    features_df = df.copy()
    if target_variable_name and target_variable_name in features_df.columns:
        features_df = features_df.drop(columns=[target_variable_name])
    
    valid_numeric_cols = [col for col in numeric_cols if col in features_df.columns]
    
    scaler: Optional[StandardScaler] = None # Initialize scaler as Optional
    if valid_numeric_cols:
        scaler = StandardScaler()
        features_df[valid_numeric_cols] = scaler.fit_transform(features_df[valid_numeric_cols])
        print(f"\nNormalized numeric columns: {valid_numeric_cols}")
    else:
        print("\nNo numeric columns found to normalize.")
        
    return features_df, scaler

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: # Changed
    """Splits data into training and testing sets, with robust stratification."""
    y_counts = Counter(y)
    min_class_count = min(y_counts.values()) if y_counts else 0
    
    stratify_param: Optional[pd.Series] = y # Explicitly type stratify_param
    
    if min_class_count < 2 : 
        print(f"\nWARNING: The least populated class in y has only {min_class_count} member(s). "
              "This is too few for stratified splitting. "
              "Falling back to non-stratified splitting.")
        stratify_param = None
    elif len(y_counts) <=1:
        print(f"\nWARNING: Only one class present in y. Cannot stratify. Falling back to non-stratified splitting.")
        stratify_param = None
    else:
        print(f"\nProceeding with stratified splitting. Minimum class count in y: {min_class_count}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    print(f"\nData split complete: {len(X_train)} train samples, {len(X_test)} test samples.")
    return X_train, X_test, y_train, y_test