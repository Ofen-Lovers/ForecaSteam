import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix

# Load your data
df = pd.read_csv('steam.csv')

# Set the target variable
y = df['Estimated owners']

# Drop the columns we’ll no longer need
cols_to_drop = [
    'AppID', 'Name', 'About the game', 'Header image',
    'Website', 'Support url', 'Support email', 'Notes',
    'Screenshots', 'Movies', 'Metacritic url', 'Reviews',
    'Publishers', 'Developers'
]
df.drop(columns=cols_to_drop, inplace=True)

# Peek at the cleaned frame
print("Shape after dropping columns:", df.shape)
print(df.head())

# Inspect missingness
missing_counts = df.isnull().sum()
missing_frac = (missing_counts / len(df)) * 100
missing_summary = pd.DataFrame({
    'missing_count': missing_counts,
    'missing_pct': missing_frac.round(2)
}).sort_values('missing_pct', ascending=False)
print("\nMissing value summary:")
print(missing_summary)

# Drop columns with too much missing data (e.g. >50%)
high_missing = missing_summary[missing_summary['missing_pct'] > 50].index.tolist()
if high_missing:
    df.drop(columns=high_missing, inplace=True)
    print(f"\nDropped columns >50% missing: {high_missing}")

# Separate features by type
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

##### Converting Categorical Variables ##### 
# Convert Release date to datetime
df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')

# Impute numeric features with median
num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Impute categorical features with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Impute Release date with earliest date
min_date = df['Release date'].min()
df['Release date'] = df['Release date'].fillna(min_date)

# Extract release month
df['Release month'] = df['Release date'].dt.month
df.drop(columns=['Release date'], inplace=True)

# Convert platform booleans to integers
df['Windows'] = df['Windows'].astype(int)
df['Mac'] = df['Mac'].astype(int)
df['Linux'] = df['Linux'].astype(int)

# Handle missing or invalid 'Categories' data
df['Categories'] = df['Categories'].apply(lambda x: [] if pd.isna(x) else x.split(';') if isinstance(x, str) else [])

# Multi-hot encode Categories using sparse matrix
mlb_cats = MultiLabelBinarizer()
cats_encoded = mlb_cats.fit_transform(df['Categories'])
cats_sparse = csr_matrix(cats_encoded)

# Add to dataframe as sparse DataFrame (removed dtype argument)
cats_df = pd.DataFrame.sparse.from_spmatrix(cats_sparse, columns=[f"Category_{c}" for c in mlb_cats.classes_], index=df.index)
df = pd.concat([df, cats_df], axis=1)
df.drop(columns=['Categories'], inplace=True)

# Handle missing or invalid 'Tags' data
df['Tags'] = df['Tags'].apply(lambda x: [] if pd.isna(x) else x.split(';') if isinstance(x, str) else [])

# Multi-hot encode Tags using sparse matrix
mlb_tags = MultiLabelBinarizer()
tags_encoded = mlb_tags.fit_transform(df['Tags'])
tags_sparse = csr_matrix(tags_encoded)

# Add to dataframe as sparse DataFrame (removed dtype argument)
tags_df = pd.DataFrame.sparse.from_spmatrix(tags_sparse, columns=[f"Tag_{t}" for t in mlb_tags.classes_], index=df.index)
df = pd.concat([df, tags_df], axis=1)
df.drop(columns=['Tags'], inplace=True)

# Handle missing or invalid 'Genres' data
df['Genres'] = df['Genres'].apply(lambda x: [] if pd.isna(x) else x.split(';') if isinstance(x, str) else [])

# Multi-hot encode Genres using sparse matrix
mlb_genres = MultiLabelBinarizer()
genres_encoded = mlb_genres.fit_transform(df['Genres'])
genres_sparse = csr_matrix(genres_encoded)

# Add to dataframe as sparse DataFrame (removed dtype argument)
genres_df = pd.DataFrame.sparse.from_spmatrix(genres_sparse, columns=[f"Genre_{g}" for g in mlb_genres.classes_], index=df.index)
df = pd.concat([df, genres_df], axis=1)
df.drop(columns=['Genres'], inplace=True)

# Handle missing or invalid 'Full audio languages' data
df['Full audio languages'] = df['Full audio languages'].apply(lambda x: [] if pd.isna(x) else x.split(';') if isinstance(x, str) else [])

# Multi-hot encode Full audio languages using sparse matrix
mlb_audio = MultiLabelBinarizer()
audio_encoded = mlb_audio.fit_transform(df['Full audio languages'])
audio_sparse = csr_matrix(audio_encoded)

# Add to dataframe as sparse DataFrame (removed dtype argument)
audio_df = pd.DataFrame.sparse.from_spmatrix(audio_sparse, columns=[f"Audio_{lang.strip()}" for lang in mlb_audio.classes_], index=df.index)
df = pd.concat([df, audio_df], axis=1)
df.drop(columns=['Full audio languages'], inplace=True)

# Handle missing or invalid 'Supported languages' data
df['Supported languages'] = df['Supported languages'].apply(lambda x: [] if pd.isna(x) else x.split(';') if isinstance(x, str) else [])

# Multi-hot encode Supported languages using sparse matrix
mlb_supported = MultiLabelBinarizer()
supported_encoded = mlb_supported.fit_transform(df['Supported languages'])
supported_sparse = csr_matrix(supported_encoded)

# Add to dataframe as sparse DataFrame (removed dtype argument)
supported_df = pd.DataFrame.sparse.from_spmatrix(supported_sparse, columns=[f"Lang_{lang.strip()}" for lang in mlb_supported.classes_], index=df.index)
df = pd.concat([df, supported_df], axis=1)
df.drop(columns=['Supported languages'], inplace=True)

# Final check
print("\nFinal shape:", df.shape)
print("Columns now:", df.columns.tolist())
print("\nTotal missing values remaining:", df.isnull().sum().sum())
